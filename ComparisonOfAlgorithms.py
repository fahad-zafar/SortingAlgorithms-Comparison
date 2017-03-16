import numpy as np
from random import randint
import matplotlib.pyplot as plt
import datetime
import time

# Global Variables for recursive Algorithms
mergeSortComparisonCount = 0
mergeSortExchangeCount = 0
deterministicComparisonCount = 0
deterministicExchangeCount = 0
randomizedComparisonCount = 0
randomizedExchangeCount = 0

# 3D array to hold time elapsed, comparisons and exchange of each algorithm
data = np.zeros((5, 3, 5))


# Insertion Sort Algorithm
def insertionSort(arr):
    comparisonCounter = 0
    exchangeCounter = 0
    for j in range(1, len(arr)):
        comparisonCounter += 1
        key = arr[j]
        i = j
        while i > 0 and key < arr[i - 1]:
            arr[i] = arr[i - 1]  # exchange happened here
            exchangeCounter += 1
            i -= 1
            comparisonCounter += 2  # in each loop iteration there are 2 comparisons

        arr[i] = key
    return comparisonCounter, exchangeCounter


# to merge small arrays
def merge(left, right, comparisonCounter, exchangeCounter):
    comparisonCounter += 1  # for the [if] statement below
    if len(left) == 0 or len(right) == 0:
        return left, comparisonCounter, exchangeCounter or right, comparisonCounter, exchangeCounter

    result = []
    i = 0
    j = 0
    while len(result) < (len(left) + len(right)):
        comparisonCounter += 1  # for the [while] loop
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

        exchangeCounter += 1  # any 1 append from above [if, else] block works
        comparisonCounter += 2  # 1 comparison takes place from above [if, else] block,
        # 2nd comparison for the if condition underneath

        if i == len(left) or j == len(right):
            result.extend(left[i:] or right[j:])  # whatever is left on left or right array, append it to result
            exchangeCounter += len(left[i:]) or len(right[j:])
            break

    return result, comparisonCounter, exchangeCounter


# Merge Sort Recursive Algorithm
def recursiveMergeSort(arr):
    global mergeSortComparisonCount, mergeSortExchangeCount

    mergeSortComparisonCount += 1  # for the [if] statement below
    if len(arr) <= 1:
        return arr, mergeSortComparisonCount, mergeSortExchangeCount

    middle = int(len(arr) / 2)
    left, mergeSortComparisonCount, mergeSortExchangeCount = recursiveMergeSort(arr[:middle])
    right, mergeSortComparisonCount, mergeSortExchangeCount = recursiveMergeSort(arr[middle:])
    result, mergeSortComparisonCount, mergeSortExchangeCount = merge \
        (left, right, mergeSortComparisonCount, mergeSortExchangeCount)
    return result, mergeSortComparisonCount, mergeSortExchangeCount


# Merge Sort Iterative Algorithm
def iterativeMergeSort(arr):
    comparisonCount = 0
    exchangeCount = 0
    lists = [[x] for x in arr]
    while len(lists) > 1:
        comparisonCount += 1
        temp = []
        for i in range(0, len(lists) // 2):
            comparisonCount += 1
            result, comparisonCount, exchangeCount = merge(lists[i * 2], lists[i * 2 + 1], comparisonCount,
                                                           exchangeCount)
            temp.append(result)
            exchangeCount += 1

        # if number of elements are odd
        comparisonCount += 1
        if len(lists) % 2:
            temp.append(lists[-1])  # append the last one as it is
            exchangeCount += 1
        lists = temp
    return lists[0], comparisonCount, exchangeCount


# Partition function used in Deterministic Quick Sort
def deterministicPartition(arr, start, end, comparisonCounter, exchangeCounter):
    pivot = arr[end]
    i = start - 1
    for j in range(start, end):
        comparisonCounter += 2  # 1 comparison for the [for] loop above, 2nd comparison for the [if] below
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            exchangeCounter += 1
    arr[i + 1], arr[end] = arr[end], arr[i + 1]
    exchangeCounter += 1
    return i + 1, comparisonCounter, exchangeCounter


# Partition function used in Randomized Quick Sort
def randomizedPartition(arr, start, end, comparisonCounter, exchangeCounter):
    randomNumber = randint(start, end)
    arr[randomNumber], arr[end] = arr[end], arr[randomNumber]
    exchangeCounter += 1
    return deterministicPartition(arr, start, end, comparisonCounter, exchangeCounter)


# Quick Sort Algorithm [Last Index as Pivot]
def deterministicQuickSort(arr, start, end):
    global deterministicComparisonCount, deterministicExchangeCount
    if start < end:
        deterministicComparisonCount += 1
        mid, deterministicComparisonCount, deterministicExchangeCount = deterministicPartition \
            (arr, start, end, deterministicComparisonCount, deterministicExchangeCount)
        deterministicComparisonCount, deterministicExchangeCount = deterministicQuickSort(arr, start, mid - 1)
        deterministicComparisonCount, deterministicExchangeCount = deterministicQuickSort(arr, mid + 1, end)
    return deterministicComparisonCount, deterministicExchangeCount


# Quick Sort Algorithm [Random Pivot]
def randomizedQuickSort(arr, start, end):
    global randomizedComparisonCount, randomizedExchangeCount
    if start < end:
        randomizedComparisonCount += 1
        mid, randomizedComparisonCount, randomizedExchangeCount = randomizedPartition \
            (arr, start, end, randomizedComparisonCount, randomizedExchangeCount)
        randomizedComparisonCount, randomizedExchangeCount = randomizedQuickSort(arr, start, mid - 1)
        randomizedComparisonCount, randomizedExchangeCount = randomizedQuickSort(arr, mid + 1, end)
    return randomizedComparisonCount, randomizedExchangeCount


def maxHeapify(A, i, size, comparisonCounter, exchangeCounter):
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i
    if left < size and A[left] > A[largest]:
        largest = left
        comparisonCounter += 2
    if right < size and A[right] > A[largest]:
        largest = right
        comparisonCounter += 2
    if largest != i:
        comparisonCounter += 1
        A[i], A[largest] = A[largest], A[i]
        exchangeCounter += 1
        comparisonCounter, exchangeCounter = maxHeapify(A, largest, size, comparisonCounter, exchangeCounter)
    return comparisonCounter, exchangeCounter


def buildMaxHeap(array, comparisonCounter, exchangeCounter):
    length = len(array)
    for i in range(length // 2, -1, -1):
        comparisonCounter += 1
        comparisonCounter, exchangeCounter = maxHeapify(array, i, len(array), comparisonCounter, exchangeCounter)
    return comparisonCounter, exchangeCounter


# Heap Sort Algorithm
def heapSort(arr):
    comparisonCounter = 0
    exchangeCounter = 0
    comparisonCounter, exchangeCounter = buildMaxHeap(arr, comparisonCounter, exchangeCounter)
    size = len(arr)
    for i in range(size - 1, 0, -1):
        comparisonCounter += 1
        arr[0], arr[i] = arr[i], arr[0]
        exchangeCounter += 1
        size -= 1
        comparisonCounter, exchangeCounter = maxHeapify(arr, 0, size, comparisonCounter, exchangeCounter)
    return comparisonCounter, exchangeCounter


# creates an array with random numbers
def generateRandomNumberedList(size):
    arr = []
    for i in range(0, 10 ** size):
        arr.append(randint(0, 2 ** 32 - 1))
    return arr


# plot graph
def plot(plt, x, index):
    plt.plot(x, data[0, index], 'm')  # recursive merge sort comparisons
    plt.plot(x, data[1, index], 'k')  # iterative merge sort comparisons
    plt.plot(x, data[2, index], 'g')  # deterministic quick sort comparisons
    plt.plot(x, data[3, index], 'b')  # randomized quick sort comparisons
    plt.plot(x, data[4, index], 'r')  # heap sort comparisons
    # plt.plot(x, data[5, index], 'c')  # insertion sort comparisons


# to write text in each plot
def writeText(y_coordinate):
    plt.text(1.0, y_coordinate, 'Merge Sort [Recursive]', rotation=90, color="m", size=7)
    plt.text(1.1, y_coordinate, 'Merge Sort [Iterative]', rotation=90, color="k", size=7)
    plt.text(1.2, y_coordinate, 'Quick Sort [Deterministic]', rotation=90, color="g", size=7)
    plt.text(1.3, y_coordinate, 'Quick Sort [Randomized]', rotation=90, color="b", size=7)
    plt.text(1.4, y_coordinate, 'Heap Sort', rotation=90, color="r", size=7)
    # plt.text(1.5, y_coordinate, 'Insertion Sort', rotation=90, color="c", size=7)


k = 0

# main program flow
for experiment in range(2, 7):
    i = 0
    j = 0
    print("\n\t\t\t\t\t\t\tInput Size\t\t\tTime Taken\t\t\tComparisons\t\t\tExchanges")

    # Merge Sort [Recursive]
    unsortedList = generateRandomNumberedList(experiment)
    previous = time.time()
    start = datetime.datetime.now().replace(microsecond=0)
    unsortedList, comparisonCount, exchangeCount = recursiveMergeSort(unsortedList)
    end = datetime.datetime.now().replace(microsecond=0)
    current = time.time()
    data[i, j, k] = current - previous
    j += 1
    data[i, j, k] = comparisonCount
    j += 1
    data[i, j, k] = exchangeCount
    print(
        "Merge Sort [Recursive]:\t\t" + str(10 ** experiment) + "\t\t\t\t" + str(end - start) + "\t\t\t\t" + str(
            comparisonCount) + "\t\t\t\t" + str(exchangeCount))
    i += 1
    j = 0

    # Merge Sort [Iterative]
    unsortedList = generateRandomNumberedList(experiment)
    previous = time.time()
    start = datetime.datetime.now().replace(microsecond=0)
    unsortedList, comparisonCount, exchangeCount = iterativeMergeSort(unsortedList)
    end = datetime.datetime.now().replace(microsecond=0)
    current = time.time()
    data[i, j, k] = current - previous
    j += 1
    data[i, j, k] = comparisonCount
    j += 1
    data[i, j, k] = exchangeCount
    print(
        "Merge Sort [Iterative]:\t\t" + str(10 ** experiment) + "\t\t\t\t" + str(end - start) + "\t\t\t\t" + str(
            comparisonCount) + "\t\t\t\t" + str(exchangeCount))
    i += 1
    j = 0

    # Quick Sort [Deterministic]
    unsortedList = generateRandomNumberedList(experiment)
    previous = time.time()
    start = datetime.datetime.now().replace(microsecond=0)
    comparisonCount, exchangeCount = deterministicQuickSort(unsortedList, 0,
                                                            len(unsortedList) - 1)
    end = datetime.datetime.now().replace(microsecond=0)
    current = time.time()
    data[i, j, k] = current - previous
    j += 1
    data[i, j, k] = comparisonCount
    j += 1
    data[i, j, k] = exchangeCount
    print("Quick Sort [Deterministic]:\t" + str(10 ** experiment) + "\t\t\t\t" + str(end - start) + "\t\t\t\t" + str(
        comparisonCount) + "\t\t\t\t" + str(
        exchangeCount))
    i += 1
    j = 0

    # Quick Sort [Randomized]
    unsortedList = generateRandomNumberedList(experiment)
    previous = time.time()
    start = datetime.datetime.now().replace(microsecond=0)
    comparisonCount, exchangeCount = randomizedQuickSort(unsortedList, 0, len(unsortedList) - 1)
    end = datetime.datetime.now().replace(microsecond=0)
    current = time.time()
    data[i, j, k] = current - previous
    j += 1
    data[i, j, k] = comparisonCount
    j += 1
    data[i, j, k] = exchangeCount
    print("Quick Sort [Randomized]:\t" + str(10 ** experiment) + "\t\t\t\t" + str(end - start) + "\t\t\t\t" + str(
        comparisonCount) + "\t\t\t\t" + str(exchangeCount))
    i += 1
    j = 0

    # Heap Sort
    unsortedList = generateRandomNumberedList(experiment)
    previous = time.time()
    start = datetime.datetime.now().replace(microsecond=0)
    comparisonCount, exchangeCount = heapSort(unsortedList)
    end = datetime.datetime.now().replace(microsecond=0)
    current = time.time()
    data[i, j, k] = current - previous
    j += 1
    data[i, j, k] = comparisonCount
    j += 1
    data[i, j, k] = exchangeCount
    print("Heap Sort:\t\t\t\t\t" + str(10 ** experiment) + "\t\t\t\t" + str(end - start) + "\t\t\t\t" + str(
        comparisonCount) + "\t\t\t\t" + str(exchangeCount))
    i += 1
    j = 0

    # Insertion Sort
    # unsortedList = generateRandomNumberedList(experiment)
    # previous = time.time()
    # start = datetime.datetime.now().replace(microsecond=0)
    # comparisonCount, exchangeCount = insertionSort(unsortedList)
    # end = datetime.datetime.now().replace(microsecond=0)
    # current = time.time()
    # data[i, j, k] = current - previous
    # j += 1
    # data[i, j, k] = comparisonCount
    # j += 1
    # data[i, j, k] = exchangeCount
    # print("Insertion Sort:\t\t\t\t" + str(10 ** experiment) + "\t\t\t\t" + str(end - start) + "\t\t\t\t" + str(
    #     comparisonCount) + "\t\t\t\t" + str(exchangeCount))

    k += 1

# labels on horizontal axis
x = [1, 2, 3, 4, 5]
labels = ['100', '1000', '10000', '100000', '1000000']

plt.figure(1)
plot(plt, x, 0)

y_coordinate = max(data[4, 0])
writeText(y_coordinate)

plt.xlabel("Input Size")
plt.ylabel("Time in Seconds")
plt.xticks(x, labels)
plt.subplots_adjust(bottom=0.15)
plt.show()

plt.figure(2)
plot(plt, x, 1)

y_coordinate = max(data[4, 1])
writeText(y_coordinate)

plt.xlabel("Input Size")
plt.ylabel("# of Comparisons")
plt.xticks(x, labels)
plt.subplots_adjust(bottom=0.15)
plt.show()

plt.figure(3)
plot(plt, x, 2)

y_coordinate = max(data[4, 2])
writeText(y_coordinate)

plt.xlabel("Input Size")
plt.ylabel("# of Exchanges")
plt.xticks(x, labels)
plt.subplots_adjust(bottom=0.15)
plt.show()