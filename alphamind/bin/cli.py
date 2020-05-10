# -*- coding: utf-8 -*-
"""
Created on 2017-6-29

@author: cheng.li
"""

import argparse
from collections import namedtuple

from sqlalchemy import create_engine

from alphamind.data.dbmodel import models
from alphamind.utilities import alpha_logger


def initdb(args):
    alpha_logger.info('DB: ' + args.url)
    engine = create_engine(args.url)
    models.Base.metadata.create_all(engine)
    alpha_logger.info('DB: initialization finished.')


Arg = namedtuple(
    'Arg', ['flags', 'help', 'action', 'default', 'nargs', 'type', 'choices', 'metavar'])
Arg.__new__.__defaults__ = (None, None, None, None, None, None, None)


class CLIFactory(object):
    args = {
        'url': Arg(
            ('-u', '--url'),
            help='set the url for the db',
            type=str)
    }

    subparsers = (
        {
            'func': initdb,
            'help': 'Initialize the metadata database',
            'args': ('url',)
        },
    )

    subparsers_dict = {sp['func'].__name__: sp for sp in subparsers}

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(
            help='sub-command help', dest='subcommand')
        subparsers.required = True

        subparser_list = cls.subparsers_dict.keys()
        for sub in subparser_list:
            sub = cls.subparsers_dict[sub]
            sp = subparsers.add_parser(sub['func'].__name__, help=sub['help'])
            for arg in sub['args']:
                arg = cls.args[arg]
                kwargs = {
                    f: getattr(arg, f)
                    for f in arg._fields if f != 'flags' and getattr(arg, f)}
                sp.add_argument(*arg.flags, **kwargs)
            sp.set_defaults(func=sub['func'])
        return parser


def get_parser():
    return CLIFactory.get_parser()
