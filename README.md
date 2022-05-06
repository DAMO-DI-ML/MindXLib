#MindXLib

## Introduction
MindXLib是达摩院决策智能实验室数据决策团队在XAI(Explainable AI，可解释机器学习)领域深耕，展示算法成果的一个open toolkit。
该算法包目前主要涵盖白盒模型相关的算法,用以解决需要可解释性的分类场景，包括rule set、rule list。

以Titanic分类问题为例，下面展示不同白盒模型的结果示例。

* rule set  
```
IF Category = 3rd class THEN Survived = No
IF Age = Adult ^ Sex 6= Female THEN Survived = No  
IF Category 6= 3rd class ^ Age 6= Adult THEN Survived = Yes
IF Category 6= 3rd class ^ Sex = Female THEN Survived = Yes
```

* rule list
```
IF Age = Adult ^ Sex 6= Female THEN Survived = No
ELSE IF Category 6= 3rd class THEN Survived = Yes
ELSE Survived = No
```

## Architecture
目前算法包主支持的模型如下：
1. pre_mining: 候选规则生成算法
    1. extract_rule: 涵盖了基于FP growth和random forest抽取rule的实现方法。
    2. wkmodes_rule：基于clustering提取候选rule
2. ruleset：当前仅支持二分类问题
    1. diver：建模为优化问题来求解
    2. drillup：基于bounded FP growth实现
    3. ruleset: 基于submodular来建模rule set learning问题，减少样本被多条rule覆盖的情况。
    4. ruleset_imb: 主要解决imbalanced datasets，以F1 score为优化目标，在保证accuracy的同时，提高recall。
3. rulelist：可支持多分类问题
    1. rulelist_SSRL

具体每个算法的使用方法请查看demo目录下的py文件。

## Related papers
1. Learning Interpretable Decision Rule Sets: A Submodular Optimization Approach