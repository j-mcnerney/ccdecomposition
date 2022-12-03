"""
A collection of general-purpose python utilities.
"""

def unique(mylist):
    """Get unique elements of L, maintaining original order of appearance."""
    return list(dict.fromkeys(mylist))

def setdiff(list1, list2):
    """Get the set difference of two lists (list1 - list 2), maintaining original order of appearance."""
    return [i for i in list1 if i not in list2]