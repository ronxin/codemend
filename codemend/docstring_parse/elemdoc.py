from recordclass import recordclass

"""
- utter: for machine to read; i.e., for model to consume
- utter_expand: for human to interpret.

# Note: these doc records can be additionally indexed by parent_id, and thus
# children can be easily found.

"""
ElemDoc = recordclass('ElemDoc', ('elem_id', 'name',
                         'full_name', 'type',
                         'parent_id', 'rtype', 'count',
                         'utter', 'utter_expand'))
