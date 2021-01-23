# Alpha - Mind

<table>
<tr>
  <td>Python version</td>
  <td><img src="https://img.shields.io/badge/python-3.6-blue.svg"/> </td>
  </tr>
<tr>
<tr>
  <td>Build Status</td>
  <td>
    <img src="https://travis-ci.org/alpha-miner/alpha-mind.svg" alt="travis build status" />
  </td>
</tr>
<tr>
  <td>Coverage</td>
  <td><img src="https://coveralls.io/repos/github/alpha-miner/alpha-mind/badge.svg?branch=master" alt="coverage" /></td>
</tr>
</table>

**Alpha - Mind** 是基于 **Python** 开发的股票多因子研究框架。

## TODO list

**alpha-mind**的开发经过长期的暂停之后，将重启。下面的列表会给出一组现在规划中的功能或者改进：

- [x] 增加对于数据后端MySQL的支持；
- [ ] 增加对于数据后端CSV文件的支持，并且提供一份样例文件供用户测试使用；
- [x] 删除所有的c++相关代码，方便alpha-mind的安装；
- [x] 在windows以及linux平台提供可以直接pip安装的安装包；
- [ ] 完整的文档；
- [ ] alpha模型增加超参数调优的功能；
- [ ] alpha模型增加多期预测能力；
- [ ] 优化器增加多期优化的能力。

## 依赖

该项目主要有两个主要的github外部依赖：

* [Finance-Python](https://github.com/alpha-miner/finance-python)

* [portfolio - optimizer](https://github.com/alpha-miner/portfolio-optimizer)：该项目是相同作者编写的用于资产组合配置的优化器工具包；

这两个库都可以直接使用pip进行安装。

## 功能

alpha - mind 提供了多因子研究中常用的工具链，包括：

* 数据清洗
* alpha 模型
* 风险模型
* 组合优化
* 执行器

所有的模块都设计了完整的测试用例以尽可能保证正确性。同时，所有的数值模型开发中都对性能给予了足够高的关注，参考了优秀的第三方工具以保证性能：

* numpy
* numba
* cvxopt
* cvxpy
* pandas
* scipy

同时还依赖于一个工具包
* [Finance-Python](https://github.com/alpha-miner/Finance-Python)

## 安装

安装需要直接clone或者下载源代码安装，具体流程为：

克隆项目到本地

```shell
$ git clone https://github.com/alpha-miner/alpha-mind.git
```

然后直接使用一下命令安装

```shell
$ python setup.py install
```

### 使用Docker运行

1. `docker build -t alpha-mind:latest -f Dockerfile .`

2. `docker run -it -p 8080:8080 --name alpha-mind alpha-mind`

#### 提示

环境变量的配置在`./entrypoint.sh`中，包括：

* `DB_VENDOR`: 如果使用mysql，请设置为`rl`;
* `DB_URI`: 数据库的连接串。
* `FACTOR_TABLES`: 使用的因子表
