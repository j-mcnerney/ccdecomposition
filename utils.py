def unique(mylist):
    """Get the unique elements of L, maintaining order of element appearance."""
    return list(dict.fromkeys(mylist))

def setdiff(list1, list2):
    """Get the set difference of two lists (list1 - list2), maintaining order of element appearance."""
    return [i for i in list1 if i not in list2]