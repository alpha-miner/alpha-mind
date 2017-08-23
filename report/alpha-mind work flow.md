# Alpha - Mind Work Flow (version 0.1.0)


## 数据表说明

* ``factors``：因子数据表；
* ``halt_list``：停牌信息；
* ``index_components``：指数成分信息；
* ``market``：行情数据；
* ``risk_exposure``：风险因子暴露；
* ``risk_cov_day``：风险模型协方差矩阵，``day``模型；
* ``risk_cov_short``：风险模型协方差矩阵，``short``模型；
* ``risk_cov_long``：风险模型协方差矩阵，``long``模型；
* ``specific_risk_day``：风险模型特质收益，``day``模型；
* ``specific_risk_short``：风险模型特质收益，``short``模型；
* ``specific_risk_long``：风险模型特质收益，``long``模型；
* ``risk_return``：风险因子收益；
* ``specific_return``：特质收益；
* ``universe``：股票池。


## 输入

### a. 测试已入库因子

* 交易日(``date``): 类型为``str``，格式：``YYYY-MM-DD``；
* 交易证券代码(``codes``)：类型为``list``，大小 $n \times 1$；
* 因子名称(``factor_names``): 类型为``list``, 大小为 $k \times 1$；
* 风险模型(``risk_model``)：类型为``str``，可选为：``day``、``short``以及``long``；
* 基准指数(``benchmark``)：用于计算超额收益，类型为``str``，例如：``zz500``代表中证500指数；
* 延迟(``delay``)：使用多少天以后的收益计算，默认为1；
* 期限(``horizon``)：在延迟``delay``天后，使用累积多少天的收益，默认为1。


### b. 测试新因子

* 交易证券代码(``codes``)：类型为``list``，大小 $n \times 1$;
* 因子数据(``factor_data``)：类型为二维数组，大小 $n \times k$;
* 风险模型(``risk_model``)：类型为``str``，可选为：``day``、``short``以及``long``;
* 基准指数(``benchmark``)：用于计算超额收益，类型为``str``，例如：``zz500``代表中证500指数；
* 延迟(``delay``)：使用多少天以后的收益计算，默认为1；
* 期限(``horizon``)：在延迟``delay``天后，使用累积多少天的收益，默认为1。

## 获取因子数据

### a. 测试已入库因子

按照``factor_names``从``factors``获取因子数据$Y$。

### b. 测试新因子

使用``factor_data``组成因子数据$Y$。

## 风险中性化

从``risk_exposure``中取出对应的风险因子暴露，对$Y$做中性化，得到$\bar Y$

## 计算预期收益

根据$\bar Y$计算得到预期收益$r$。

## 计算股票协方差矩阵

根据对应的``risk_model``，从``risk_cov_*``表中取出对应的风险因子协方差矩阵，根据对应的``risk_exposure``以及``specific_risk_*``计算得到股票协方差矩阵$C$：

$$ C = E^T \times RC \times E + SR $$

其中:

* $E$为股票的``risk_exposure``;
* $RC$为风险因子的协方差矩阵；
* $SR$为股票的``specific_risk``。

## 计算组合

根据预期收益$r$和预期协方差矩阵$C$，计算最优组合$\omega$
