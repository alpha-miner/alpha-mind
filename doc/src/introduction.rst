************
入门介绍
************

在这篇很短的入门介绍中，我们将为您介绍使用多因子建模框架完成数据导入、因子挖掘、因子组合、组合优化
以及回测的全流程。


流程图
===============

略

数据接入
===============

多因子框架有自带的数据表结构需求，现阶段可以对接mysql以及postgresql（未来会接入更多的
数据库，例如：sqlserver）。数据表结构对于用户是透明的，用户在指定数据源的时候，只需要使用如下的语句：

.. code-block:: py
    :linenos:

    from alphamind.api import *

    data_source = "url_for_some_database"
    engine = SqlEngine(data_source)

回测设置
===============

回测阶段，可以做各种配置，例如:

* 起始时间
* 调仓周期
* 股票池，一般为某个指数成分股
* 行业分类，现在支持申万行业，代码为"sw";
* 基准指数
* 组合方法

在代码中，可以表示为：

.. code-block:: py
    :linenos:

    start_date = '2020-01-01'
    end_date = '2020-02-21'

    freq = '10b'
    industry_name = 'sw'
    universe = Universe('hs300')
    benchmark_code = 300
    method = 'risk_neutral'

因子池
====================

用户可以定义完整的因子池，多因子框架支持任意多个因子的回测：

.. code-block:: py
    :linenos:

    alpha_factors = {
    'f01': CSQuantiles(LAST('EMA5D')),
    'f02': CSQuantiles(LAST('EMV6D')),
    }

这里面，我们使用了EMA50和EMV6D两个因子，并且都对他们做了分位数化。

机器学习模型
=====================

为了将因子组合起来，我们会搭载一个alpha模型：

.. code-block:: py
    :linenos:

    weights = dict(f01=1., f02=1.)
    alpha_model = ConstLinearModel(features=alpha_factors, weights=weights)

这里我们使用了一个等权加权模型。多因子框架，支持多款不同的机器学习模型，用户也可以接入自己自定义的模型。

组合优化
=====================

多因子框架中，组合优化的基本原理是mean-variance优化， 但是支持很多特性：

* 总杠杆率约束；
* 行业权重约束；
* 风格因子约束；
* 换手率约束；
* 成分股权重限制；

.. code-block:: py
    :linenos:

    # Constraintes settings

    industry_names = industry_list(industry_name, industry_level)
    constraint_risk = ['SIZE', 'SIZENL', 'BETA']
    total_risk_names = constraint_risk + ['benchmark', 'total']
    all_styles = risk_styles + industry_styles + macro_styles

    b_type = []
    l_val = []
    u_val = []

    previous_pos = pd.DataFrame()
    rets = []
    turn_overs = []
    leverags = []

    for name in total_risk_names:
        if name == 'benchmark':
            b_type.append(BoundaryType.RELATIVE)
            l_val.append(0.8)
            u_val.append(1.0)
        else:
            b_type.append(BoundaryType.ABSOLUTE)
            l_val.append(0.0)
            u_val.append(0.0)

    bounds = create_box_bounds(total_risk_names, b_type, l_val, u_val)
    turn_over_target = 0.4

上面的代码，使得：

* 成分股权重不低于80%；
* 总权重为100%（无杠杆和现金保留）
* 在SIZE，SIZENL以及BETA三个风格因子上，相对于基准无暴露;
* 单次换手不超过40%（双边计算）

将一切组合起来...
===========================

通过简单的调用内置函数，就可以完成完整的回测：

.. code-block:: py
    :linenos:

    running_setting = RunningSetting(weights_bandwidth=weights_bandwidth,
                                     rebalance_method=method,
                                     bounds=bounds,
                                     turn_over_target=turn_over_target)
    
    # Strategy
    strategy = Strategy(alpha_model,
                        data_meta,
                        universe=universe,
                        start_date=start_date,
                        end_date=end_date,
                        freq=freq,
                        benchmark=benchmark_code)

    strategy.prepare_backtest_data()
    ret_df, positions = strategy.run(running_setting=running_setting)


画图
===============

上一步返回的`ret_df`包含具体的收益信息，`positions`包含完整的持仓记录。用户可以自行绘图
查看结果，例如：

.. code-block:: py
    :linenos:

    ret_df[['turn_over', 'excess_return']].cumsum().plot(figsize=(14, 7), secondary_y='turn_over')

将累计超额收益以及累积换手绘制出来。

完整的例子
=================

完整的代码可以在notbook文件夹下，例子：Example 2 - Strategy Analysis.ipynb
