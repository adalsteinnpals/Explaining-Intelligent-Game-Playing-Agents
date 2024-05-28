import operator
from collections import defaultdict

class Variable:
    def __init__(self, name: str, domain: list, status=False, value=None):
        self.name = name
        self.status = status # false not set true set
        self.domain = list(domain)
        self.original_domain = list(domain)
        self.value = value
        self.history = []

    def set_value(self, value):
        if value not in self.domain:
            raise Exception("setting value outside of domain")
        if self.status:
            raise Exception("setting value on already set variable")
        self.history.append(self.domain[:])
        self.domain = [value]
        self.value = value
        self.status = True

    def unset_value(self):
        if not self.status:
            raise Exception("unsetting value on unset variable")
        self.status = False
        self.undo_domain_change()

    def undo_domain_change(self):
        self.domain = self.history.pop(-1)[:]

    def add_to_domain(self, value):
        self.domain.append(value)

    def remove_from_domain(self, value):
        self.domain.remove(value)

    def bulk_remove_from_domain(self, values: list):
        for v in values:
            self.domain.remove(v)

    def get_name(self):
        return self.name

    def get_domain(self):
        return self.domain

    def get_original_domain(self):
        return self.original_domain

    def __repr__(self):
        return "Variable({},{},{},{})".format(self.name, self.domain, self.status, self.value)

    def __hash__(self):
        return str.__hash__(repr(self))

    def __str__(self):
        out_str ="Variable:\n\tname: {}\n\tstatus: {}".format(self.name, self.status)
        if self.status:
            out_str += "\n\tvalue: {}".format(self.value)
        else:
            domain = '{' + ','.join([str(v) for v in self.domain]) + '}'
            out_str += "\n\tdomain: {}".format(domain)
        return out_str

    """***************************************
        COMPARATORS START!
    """
    def __eq__(self, other):
        if self.get_name() == other.get_name():
            return True
        if not self.status or not other.status:
            return True
        return self.value == other.value


    def __ne__(self, other):
        if not self.status or not other.status:
            return True
        if self.get_name() == other.get_name():
            return False
        return self.value != other.value

    def __lt__(self, other):
        if not self.status or not other.status:
            return True
        return self.value < other.value

    def __gt__(self, other):
        if not self.status or not other.status:
            return True
        return self.value > other.value

    def __le__(self, other):
        if not self.status or not other.status:
            return True
        return self.value <= other.value

    def __ge__(self, other):
        if not self.status or not other.status:
            return True
        return self.value >= other.value
    """**************************************
        COMPARATORS END!
    """
