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

* [portfolio - optimizer](https://github.com/alpha-miner/portfolio-optimizer)：该项目是相同作者编写的用于资产组合配置的优化器工具包；

* [xgboost](https://github.com/dmlc/xgboost)： 该项目是alpha - mind中一些机器模型的基础库。

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

## 数据源

我们在工具包中也提供了一个数据源的参考实现。这个数据源的参考实现来自于``通联数据``提供的股票因子数据和风险模型数据等，具体细节可以参考：[优矿](https://uqer.io)。

该数据源使用RMDBS，供参考的数据库可以是Postgresql。在工具包中我们已经提供了命令行工具，帮助用户一键配置好数据库。步骤如下：

下面的步骤以Ubuntun上Postgresql为例子：

* 安装数据库软件
  
  请前往[PostgreSQL官网](https://www.postgresql.org/)，根据指导，下载安装PostgreSQL数据库。

* 新建数据库

  在安装完成的数据库中新建``Database``，例如名字：``alpha``。**注意这个数据需要使用``utf8``作为编码**。

* 一键配置数据库

  在命令行中运行：

  ```bash
  alphadmind initdb --url postgresql+psycopg2://user:pwd@host/alpha
  ```

  其中：

  * ``user``：数据库用户名
  * ``pwd``：用户密码
  * ``host``：数据库服务器地址

  如果成功，会有类似的输出：

  ```
  2017-06-29 14:48:36,678 - ALPHA_MIND - INFO - DB: postgresql+psycopg2://user:pwd@host/alpha
  2017-06-29 14:48:37,515 - ALPHA_MIND - INFO - DB: initialization finished.
  ```

* Windows

  对于Windows使用者，命令行工具alphamind并不能直接使用，这个时候可以使用变通的办法，进入源码alphamind/bin目录下：

  ```bash
  python alphadmind initdb --url postgresql+psycopg2://user:pwd@host/alpha
  ```
  
  可以达到一样的效果。
  
* 数据库更新

  在目录``scripts`` 下有[airflow]()脚本文件``update_uqer_data.py``可以用来做每天的数据更新。使用之前除了要配置好airflow服务器之外，需要更新脚本中以下两行：

  ```
  _ = uqer.Client(token='')
  engine = sqlalchemy.create_engine('')
  ```

  其中token需要填入有效的通联数据认证信息；engine需要填入上面指定的数据库地址。

