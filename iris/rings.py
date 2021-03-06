"""One Ring to rule them all, two ring to get a degree."""


def merge_ring_and_parameters(ring, parameters):
    labels = ring.values()
    return {key: value for key, value in zip(labels, parameters)}


W1 = {
    0: 'Z4',
    1: 'Z9',
    2: 'Z16',
    3: 'Z25',
}

W2 = {
    0: 'Z4',
    1: 'Z5',
    2: 'Z6',
    3: 'Z7',
    4: 'Z8',
    5: 'Z9',
    6: 'Z12',
    7: 'Z13',
    8: 'Z14',
    9: 'Z15',
    10: 'Z16',
    11: 'Z21',
    12: 'Z22',
    13: 'Z23',
    14: 'Z24',
    15: 'Z25',
}

W3 = {
    0: 'Z4',
    1: 'Z5',
    2: 'Z6',
    3: 'Z7',
    4: 'Z8',
    5: 'Z9',
    6: 'Z12',
    7: 'Z13',
    8: 'Z14',
    9: 'Z15',
    10: 'Z16',
    11: 'Z21',
    12: 'Z22',
    13: 'Z23',
    14: 'Z24',
    15: 'Z25',
    16: 'Z10',
    17: 'Z11',
}
