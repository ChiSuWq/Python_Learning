"""
filename: Argparse_Learning.py

@author:Su_Chi
@date: 2019/8/13 16:12
"""

import argparse


"""
argparse.ArgumentParser([description][, epilog][, prog][, usage][, add_help][, argument_default][, parents]
[, prefix_chars][,conflict_handler][, formatter_class])

创建一个ArgumentParser对象。每个参数都有其特殊含义，简单的介绍如下：

description：help参数之前显示的信息

epilog：help参数之后显示的信息。

add_help:给解析器添加-h/--help选项（默认为True）

argument_default:设置参数的默认值（默认为None）

parents:ArgumentParser对象组成列表，这些对象中的参数也要包含进来。

prefix_chars：可选参数之前的前缀（默认为-)

fromfile_prefix_chars:如果是从文件中读取参数，这个文件名参数的前缀（默认为None）

formatter_class:一个自定义帮助信息格式化输出的类

conflict_handler:通常不需要，定义了处理冲突选项的策略

prog：程序名（默认为sys.argv[0]）

usage:程序的使用用例，默认情况下会自动生成
"""

#description 在help参数之前显示的信息，通常简短的描述这个程序的用途
parser1 = argparse.ArgumentParser(description= 'A foo that bars')
print(parser1.print_help())

#epilog(收场白) 在help参数之后显示的信息
parser2 = argparse.ArgumentParser(description= 'A foo that bars', epilog= "And that's how you'd foo a bar" )
print(parser2.print_help())

#add_help() 默认情况下，ArgumentParser对象自动添加-h/--help选项，以展示帮助信息
#通过设置add_help= False可以取消帮助信息的显示
parser3 = argparse.ArgumentParser()
parser3.add_argument('--foo', help= 'foo help')
print(parser3.print_help())

parser4 = argparse.ArgumentParser(prog='PROG', add_help=False)
parser4.add_argument('--foo', help='foo help')
print(parser4.print_help())#则print中不再输出 -h/--help信息

#prefix_chars 前缀,大多数命令行使用-作为前缀， prefix_char=argument可以自定义前缀
parser5 = argparse.ArgumentParser(prog='PROG', prefix_chars='-+')
parser5.add_argument('+f')
parser5.add_argument('++bar')
print('\n', parser5.parse_args('+f X ++bar Y'.split()))
print(parser5.parse_args(['+f','x','++bar','y']))

#fromfile_prefix_chars 定义一个从文件中获取参数的前缀,解析器解析带有该前缀的文件名
with open('abc.txt', 'w') as fp:
	fp.write('-f\nbar')
parser6 = argparse.ArgumentParser(prog='fromfile_prefix_chars_',fromfile_prefix_chars='@')
parser6.add_argument('-f', help='foo help')
print(parser6.parse_args(['-f', 'foo', '@abc.txt']))

#argument_default 设置参数的默认值，如果是默认参数，则所有实参都必须有默认值；如果是argparse.SUPPRESS，则实参不一定要有默认值
parser7 = argparse.ArgumentParser(prog='Argument_Default_', argument_default= argparse.SUPPRESS)#
parser7.add_argument('-f', dest='f_dest')
parser7.add_argument('bar', nargs='?')
print('parser_withAD:', parser7.parse_args(['-f', 'foo']))
print('parser_withAD:', parser7.parse_args([]))

parser8 = argparse.ArgumentParser(prog='Argument_Default_')
parser8.add_argument('-f', dest='f_dest')
parser8.add_argument('bar', nargs='?')
print('parser_noAD:',parser8.parse_args(['-f', 'foo']))
print('parser_noAD:',parser8.parse_args([]))
"""
1、
如果argument_default= argparse.SUPPRESS（意为抑制实参的默认值）
则 parse_args([])  return的Namespace中没有实参f和bar 即return Namespace()

若Parameter:argument_default为默认值，则所有实参有默认值,则实参f和bar必然存在
即return Namespace(f=None, bar=None)

2、add_argumnet中的Parameter:'dest'
如果add_argument中有dest,则生成的Namespace中对应的argument名为dest的变量。

如parser7中的-f dest='f_dest', 则生成Namespace(f_dest=...)
'f' in flags 返回False

"""

flags=None
flags = parser7.parse_args(['bar', '-f', 'foo'])
#print(type(flags))
print(flags)
print('f' in flags)
print('f_dest' in flags)



