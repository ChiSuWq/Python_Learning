# Module Argparse

* 1  
如果argument_default= `argparse.SUPPRESS`（意为抑制实参的默认值）  
则 parse_args([])  return的Namespace中没有实参f和bar 即return Namespace()  
若Parameter:argument_default为默认值，则所有实参有默认值,则实参f和bar必然存在
即return Namespace(f=None, bar=None)

  |argument_default | return |'f' in flags|
  |:-------:|:-------:|:-------:|
  |SUPPRESS| Namespace()|False|
  |DEFAULT| Namespace(f=None, bar=None)|Ture|

* 2  
add_argumnet中的Parameter:`dest`
如果add_argument中有`dest` ,则生成的Namespace中对应的argument名改为dest形参的值。  
如parser7中的-f dest='f_dest', 则生成Namespace(f_dest=...)
'f' in flags 返回False

  |dest| return |'f' in flags|
  |:-------:|:-------:|:-------:|
  |DEFAULT| Namespace(f=...)|False|
  |f_dest| Namespace(f_dest=...)|Ture|
