#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/graph_executor_impl.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <stack>

namespace torch {
namespace jit {

namespace {

class AttributePropagator {
 private:
  // Contains attributes that can't be folded or user directs to keep them.
  std::unordered_map<script::ModulePtr, std::set<std::string>> preservedAttrs_;

  // findConstantAttr function locates the sub Module where attributes are
  // defined. The algorithm chases getAttr chains to locate the submodules.
  // For example:
  // module M {
  //   attributes {
  //     A = <SubModule at ...>
  //   }
  //   ...
  //   %A = prim::GetAttr[name="A"](%self)
  //   ...
  //   %B = prim::GetAttr[name="B"](%A)
  //   ...
  //   %weight = prim::GetAttr[name="scale"](%B)
  //   ...
  //   submodules {
  //     module SubModule {
  //       attributes {
  //          B = <SubModule2 at ...>
  //       }
  //       submodules {
  //         module SubModule2 {
  //            attributes {
  //               scale = 2
  //            }
  //         }
  //       }
  //     }
  //   }
  //
  // findConstantAttr(%B, "scale", M)  returns true because there are no
  // explicit SetAttr that modifies %B. attr_module points to the module where
  // attribute lives (in this example it is <SubModule2 at ...>).
  //
  // Note inplace mutations to attributes are checked later using alias
  // analysis.
  //
  // We can use a more efficient algorithm to hash each constant GetAttr to its
  // corresponding value. Based on initial test on resnet50 and other torch
  // vision tests. GetAttrs are not too frequent so it is ok to chase GetAttr
  // chain to retrieve their values.
  bool findConstantAttr(
      Node* input,
      std::string& name,
      script::Module& attr_module) {
    std::stack<std::string> names;
    while (!(input->outputs()[0]->type() == attr_module.type())) {
      if (input->kind() == prim::GetAttr) {
        names.push(input->s(attr::name));
        input = input->inputs()[0]->node();
      } else {
        return false;
      }
    }

    while (!names.empty()) {
      auto m_name = names.top();
      names.pop();
      attr_module = attr_module.attr(m_name).toModule();
      auto it = preservedAttrs_.find(attr_module._ivalue());
      if (it != preservedAttrs_.end()) {
        if (it->second.count(m_name)) {
          return false;
        }
      }
    }
    auto it = preservedAttrs_.find(attr_module._ivalue());
    return it == preservedAttrs_.end() || !it->second.count(name);
  }

  void recordMutableAttrs(
      std::shared_ptr<Graph>& graph,
      script::Module& module) {
    std::stack<Block*> blocks({graph->block()});
    std::unique_ptr<AliasDb> aliasDb =
        torch::make_unique<AliasDb>(graph, /* isFrozen */ true);
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::SetAttr || n->kind() == prim::GetAttr) {
          auto name = n->s(attr::name);
          auto input = n->inputs()[0]->node();
          auto attr_module = module;
          if (!findConstantAttr(input, name, attr_module)) {
            continue;
          }
          if (n->kind() == prim::SetAttr || aliasDb->hasOutputWriters(n)) {
            GRAPH_DEBUG(
                n->kind() == prim::GetAttr ? "attribute: " + name + " in %" +
                        n->outputs()[0]->debugName() + " has inplace writer"
                                           : "");
            preservedAttrs_[attr_module._ivalue()].insert(name);
          }
        }
      }
    }
  }

  std::set<std::string> getReferencedAttrs(
      std::shared_ptr<Graph>& graph,
      script::Module& module_clone) {
    std::stack<Block*> blocks({graph->block()});
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto n : block->nodes()) {
        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::GetAttr) {
          auto& name = n->s(attr::name);
          if (module_clone.hasattr(name)) {
            preservedAttrs_[module_clone._ivalue()].insert(name);
          }
        }
      }
    }
    auto it = preservedAttrs_.find(module_clone._ivalue());
    if (it != preservedAttrs_.end())
      return it->second;
    else
      return std::set<std::string>();
  }

  bool shouldFoldAttr(const IValue& attr, std::string& name, bool is_eval) {
    if (attr.isTensor()) {
      auto t = attr.toTensor();
      if (t.requires_grad()) {
        if (is_eval) {
          t.set_requires_grad(false);
        } else {
          return false;
        }
      }
    }
    // Do not fold training attribute in training mode.
    return is_eval || name != "training";
  }

 public:
  void propagateAttributes(
      std::shared_ptr<Graph>& graph,
      script::Module& module_clone) {
    std::unordered_map<
        script::ModulePtr,
        std::unordered_map<std::string, Value*>>
        attrValues;
    auto is_eval = !module_clone.is_training();
    GRAPH_DEBUG("Freezing Module in ", is_eval ? "eval mode" : "training mode");
    auto block = graph->block();
    std::stack<Block*> blocks({block});
    // Record Attributes that are explicitely set in the module. They cannot be
    // folded.
    recordMutableAttrs(graph, module_clone);
    Node* m = *block->nodes().begin();
    WithInsertPoint guard(m);
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        Node* n = *it;
        it++; // advance iterator bc the current node may be destroyed

        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::GetAttr) {
          auto name = n->s(attr::name);
          auto attr_module = module_clone;
          auto input = n->inputs()[0]->node();
          if (!findConstantAttr(input, name, attr_module)) {
            GRAPH_DEBUG("attribute: ", name, " is mutable.")
            continue;
          }
          assert(attr_module.hasattr(name));
          Value* param_const = nullptr;
          auto I = attrValues.find(attr_module._ivalue());
          if (I != attrValues.end()) {
            auto II = I->second.find(name);
            if (II != I->second.end())
              param_const = II->second;
          }
          if (!param_const) {
            auto attr = attr_module.attr(name);
            if (!shouldFoldAttr(attr, name, is_eval))
              continue;
            if (auto attrVal = tryInsertConstant(*graph, attr)) {
              param_const = *attrVal;
            } else {
              GRAPH_DEBUG(
                  attr.type()->cast<ClassType>() ? "" : "attribute: ",
                  name,
                  " is not materializable.");
              continue;
            }
            auto m_name = attr_module.type()->name()->qualifiedName();
            m_name += ".";
            m_name += name;
            param_const->setDebugName(m_name);
            attrValues[attr_module._ivalue()][name] = param_const;
          }
          GRAPH_UPDATE(
              "Folding GetAttr %",
              n->outputs()[0]->debugName(),
              " with ",
              param_const->debugName());
          n->outputs().at(0)->replaceAllUsesWith(param_const);
          n->removeAllInputs();
          n->destroy();
        }
      }
    }
  }

  // cleanupFrozenModule function cleans up the Frozen module. it performs the
  // following:
  // 1) Remove unused attributes.
  // 2) Remove unreferenced submodules
  // 3) Remove non pulic unreferenced methods.
  // TODO: do #3 because there is no API to 'unsafely' remove methods.
  void cleanupFrozenModule(
      std::shared_ptr<Graph>& graph,
      script::Module& module_clone) {
    std::vector<std::string> attrsToRemove;
    auto type = module_clone.type();
    size_t N = type->numAttributes();
    auto KeepAttrs = getReferencedAttrs(graph, module_clone);
    for (size_t i = 0; i < N; ++i) {
      auto attrTy = type->getAttribute(i);
      auto name = type->getAttributeName(i);
      if (!KeepAttrs.count(name)) {
        attrsToRemove.push_back(name);
      }
    }
    for (auto& name : attrsToRemove) {
      module_clone._ivalue()->unsafeRemoveAttr(name);
      module_clone.type()->unsafeRemoveAttribute(name);
    }
  }
}; // class AttributePropagator
} // namespace

script::Module freezeModule(const script::Module& module) {
  AttributePropagator attrPropagator;
  auto module_clone = module.clone();
  script::Method method = module_clone.get_method("forward");
  auto graph = method.graph();
  Inline(*graph);
  attrPropagator.propagateAttributes(graph, module_clone);
  runOptimization(graph, /* unroll? */ false);
  attrPropagator.cleanupFrozenModule(graph, module_clone);

  GRAPH_DUMP(
      module_clone.type()->name()->name() + "::forward() after freezing module",
      method.graph());
  return module_clone;
}

} // namespace jit
} // namespace torch
