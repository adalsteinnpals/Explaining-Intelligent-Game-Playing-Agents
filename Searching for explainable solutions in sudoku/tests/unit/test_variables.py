import pytest

from CSP.variable import Variable


def test_construct_variables():
    domain = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    v1 = Variable('A1', domain)
    v2 = Variable('A2', [1], True, 1)
    assert v1.get_domain() == domain
    assert not v1.status
    assert v2.get_domain() == [1]
    assert v2.status


def test_variable_set():
    domain = [1, 2, 3, 4]
    v1 = Variable('A1', domain)
    v2 = Variable('A2', domain)

    assert not v1.status
    assert v1.get_domain() == domain

    v1.set_value(1)

    assert v1.get_domain() == [1]
    assert v1.status

    with pytest.raises(Exception) as e:
        assert v2.set_value(99)
    assert str(e.value) == "setting value outside of domain"


def test_variable_unset():
    domain = [1, 2, 3, 4]
    v1 = Variable('A1', domain)
    assert not v1.status
    assert v1.get_domain() == domain

    v1.set_value(1)

    assert v1.get_domain() == [1]
    assert v1.status

    v1.unset_value()

    assert v1.get_domain() == domain
    assert not v1.status
