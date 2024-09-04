function booleanAnd(arr1, arr2) {
    return arr1.map((value, index) => value && arr2[index]);
}

function booleanOr(arr1, arr2) {
    return arr1.map((value, index) => value || arr2[index]);
}

function findMin(a, b) {
    if (a<=b) {
        return a;
    }
    return b;
}

function findMax(a, b) {
    if (a>=b) {
        return a;
    }
    return b;
}
