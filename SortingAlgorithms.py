from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from itertools import accumulate
from typing import List, Tuple, Generator, Type, Optional, Callable,Union
from math import log10,floor, ceil, log2

try:
    from typing import Self
except ImportError:
    ...




class Action(Enum):
    COMPARE:int = 0
    SWAP:int = 1

SortingAction = Tuple[Action,int,int, Optional[Union[int,Tuple[int,int]]]]

class SortingAlgorithm(ABC):

    def __init__(this, data:List[int]) -> None:
        this._data = data
        this._comparisons = 0
        this._swaps = 0

    def __len__(this) -> int:
        return len(this._data)

    def __getitem__(this,idx:int) -> int:
        return this._data[idx]

    def compare(this,i1:int,i2:int) -> int:
        assert 0 <= i1 < len(this)
        assert 0 <= i2 < len(this)

        this._comparisons+=1

        if (this[i1] == this[i2]): return 0

        return 1 if (this[i1] > this[i2]) else -1

    def swap(this,i1:int,i2:int) -> None:
        this._data[i1], this._data[i2] = this._data[i2], this._data[i1]
        this._swaps +=1

    @property
    def number_comparisons(this) -> int:
        return this._comparisons

    @property
    def number_swaps(this) -> int:
        return this._swaps

    @abstractmethod
    def make_step(this) -> Generator[SortingAction,None,None]:
        ...

    def index(this, value:int):
        return this._data.index(value)


class BubbleSort(SortingAlgorithm):

    def make_step(this) -> Generator[SortingAction,None,None]:

        n = len(this)

        for i in range(n):

            swapped = False

            for j in range(0,n-i-1):
                cmp = this.compare(j,j+1)
                yield (Action.COMPARE,j,j+1, None)

                if cmp>0:
                    this.swap(j,j+1)
                    yield (Action.SWAP, j, j + 1, None)
                    swapped = True

            if (not swapped):
                break

class InsertionSort(SortingAlgorithm):

    def make_step(this) -> Generator[SortingAction,None,None]:
        n = len(this)

        for i in range(n-1):
            j=i+1

            while (j>0):
                cmp = this.compare(j-1,j)
                yield (Action.COMPARE,j-1,j, None)

                if (cmp>0):
                    this.swap(j-1,j)
                    yield (Action.SWAP, j, j - 1, None)
                    j-=1
                else:
                    break

class SelectionSort(SortingAlgorithm):

    def make_step(this) -> Generator[SortingAction,None,None]:
        n = len(this)

        for i in range(n-1):
            min_idx = i

            for j in range(i+1,n):
                cmp = this.compare(min_idx,j)
                yield (Action.COMPARE,min_idx, j, None)

                if (cmp>0):
                    min_idx = j

            if (min_idx != i):
                this.swap(min_idx,i)
                yield (Action.SWAP, min_idx, i, None)


class CombSort(SortingAlgorithm):

    def __init__(this, shrink_factor:float = 1.3, *args, **kwargs):
        super().__init__(*args,**kwargs)
        this._shrink_factor = shrink_factor

    def make_step(this):
        n = len(this)
        gap = n
        finished = False

        while (not finished):
            gap = int(gap / this._shrink_factor)

            if (gap<=1):
                gap = 1
                finished = True
            elif 9<=gap<=10:
                gap = 11 #rule of 11

            i = 0

            while ((i+gap) < n):
                cmp = this.compare(i,i+gap)
                yield (Action.COMPARE,i,i+gap, None)

                if (cmp>0):
                    this.swap(i,i+gap)
                    yield (Action.SWAP,i,i+gap, None)
                    finished = False

                i+=1


class QuickSort(SortingAlgorithm):

    def compare(this,i1:int,pivot:int) -> int:
        assert 0 <= i1 < len(this)

        this._comparisons+=1

        if (this[i1] == pivot): return 0

        return 1 if (this[i1] > pivot) else -1

    def _partition(this, low,high) -> Generator[SortingAction,None, Tuple[int, int]] :
        pivot = this[(low+high) // 2]

        lt = low
        eq = low
        gt = high

        while eq <= gt:
            cmp = this.compare(eq, pivot)
            yield (Action.COMPARE,eq,this.index(pivot), (low,high))

            if (cmp<0):
                if (eq != lt):
                    this.swap(eq, lt)
                    yield (Action.SWAP,eq, lt, (low,high))

                lt+=1
                eq+=1
            elif (cmp>0):
                if (eq!=gt):
                    this.swap(eq, gt)
                    yield (Action.SWAP, eq, gt, (low,high))

                gt-=1
            else:
                eq+=1

        return (lt,gt)



    def _quicksort(this,low, high) -> Generator[SortingAction,None,None]:
        if ((low>=0) and (high>=0) and (low<high)):

            lt,gt = yield from this._partition(low,high)

            yield from this._quicksort(low,lt-1)
            yield from this._quicksort(gt+1, high)

    def make_step(this) -> Generator[SortingAction,None,None]:

        yield from this._quicksort(0,len(this)-1)


#Implementation: https://www.geeksforgeeks.org/in-place-merge-sort/
class MergeSort(SortingAlgorithm):

    def _inplace_merge(this, a:int, b:int) -> Generator[SortingAction,None,None]:

        calculate_gap = lambda g: 0 if g<=1 else int(ceil(g/2))

        gap = b-a+1
        gap = calculate_gap(gap)

        while (gap>0):
            i=a

            while ((i+gap) <=b):
                j=i+gap

                cmp = this.compare(i,j)
                yield (Action.COMPARE,i,j, (a,b))

                if (cmp>0):
                    this.swap(i,j)
                    yield (Action.SWAP,i,j, (a,b))

                i+=1

            gap = calculate_gap(gap)

    def _mergesort(this, a:int, b:int) -> Generator[SortingAction,None,None]:
        if (a==b):
            return

        m = (a+b)//2

        yield from this._mergesort(a,m)
        yield from this._mergesort(m+1, b)

        yield from this._inplace_merge(a,b)

    def make_step(this) -> Generator[SortingAction,None,None]:
        yield from this._mergesort(0,len(this)-1)


#Implementation: https://www.geeksforgeeks.org/heap-sort/
class HeapSort(SortingAlgorithm):

    def _heapify(this, length, idx):
        largest = idx
        l = 2 * idx + 1
        r = 2 * idx + 2

        area = (idx,r)


        if (l<length):
            cmp = this.compare(largest,l)
            yield (Action.COMPARE,largest,l, area)

            if (cmp<0):
                largest = l


        if (r<length):
            cmp = this.compare(largest, r)
            yield (Action.COMPARE, largest, r, area)

            if (cmp<0):
                largest = r

        if (largest != idx):
            this.swap(largest,idx)
            yield (Action.SWAP, largest, idx, area)

            yield from this._heapify(length, largest)



    def make_step(this) -> Generator[SortingAction,None,None]:
        n = len(this)

        for i in range(n//2 -1, -1, -1):
            yield from this._heapify(n,i)

        for i in range(n-1,0,-1):
           this.swap(0,i)
           yield (Action.SWAP,0,i, None)
           yield from this._heapify(i,0)


#Implementation https://www.geeksforgeeks.org/timsort/
class TimSort(MergeSort):
    MIN_MERGE = 32

    @staticmethod
    def calcMinRun(n):

        r=0

        while n >= TimSort.MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def make_step(this) -> Generator[SortingAction,None,None]:
        n = len(this)

        minRun = TimSort.calcMinRun(n)

        for start in range(0, n, minRun):
            end = min(start + minRun - 1, n - 1)

            tmp = [this[x] for x in range(start,end+1)]

            ins_sort = InsertionSort(tmp)

            for (action,ii,jj,_) in ins_sort.make_step():
                i = this.index(ins_sort[ii])
                j = this.index(ins_sort[jj])

                yield (action, i, j, None)


                if action == Action.SWAP:
                    this.swap(i,j)
                elif action == Action.COMPARE:
                    this._comparisons+=1



        size = minRun

        while size < n:
            for left in range(0,n,2*size):
                mid = min(n - 1, left + size - 1)
                right = min((left + 2 * size - 1), (n - 1))

                if mid < right:

                    yield from this._inplace_merge(left,right)

            size *= 2


# https://panthema.net/2013/sound-of-sorting/sound-of-sorting-0.6.5/src/SortAlgo.cpp.html
class MSDRadixSort(SortingAlgorithm):

    def __init__(this, lsd=False,*args,**kwargs):

        super().__init__(*args,**kwargs)

        this._lsd = lsd

    def _get_digit(this, index:int, d:int) -> int:
        return (this[index] // (10 ** d)) % 10

    def _radix_sort(this, lo:int, hi:int, depth:int) -> Generator[SortingAction,None,None]:

        pmax = floor(log10( max(this._data) + 1))
        assert depth <= pmax


        # count digits
        count = [0] * 10

        for i in range(lo, hi):
            r = this._get_digit(i,pmax-depth)
            assert r < 10
            count[r] += 1

        # inclusive prefix sum
        bkt = list(accumulate(count))


        i = 0
        while i < (hi - lo):
            while True:
                r = this._get_digit(lo+i,pmax-depth)
                j = bkt[r] - 1
                bkt[r] -= 1
                if j <= i:
                    break
                this.swap(lo + i, lo + j)
                yield (Action.SWAP,lo + i, lo + j,None)
            i += count[this._get_digit(lo+i,pmax-depth)]


        # no more depth to sort?
        if depth + 1 > pmax:
            return

        # recurse on buckets
        sum_lo = lo
        for i in range(10):
            if count[i] > 1:
                yield from this._radix_sort(sum_lo, sum_lo + count[i], depth + 1)
                sum_lo += count[i]


    def make_step(this) -> Generator[SortingAction,None,None]:

        yield from this._radix_sort(0,len(this),0)




#https://arxiv.org/pdf/2110.01111

class ICantBelieveItCanSort(SortingAlgorithm):

    def make_step(this) -> Generator[SortingAction,None,None]:

        n = len(this)

        for i in range(1,n):
            for j in range(i):
                cmp = this.compare(i,j)
                yield (Action.COMPARE,i,j, None)

                if (cmp<0):
                    this.swap(i,j)
                    yield (Action.SWAP,i,j, None)


class StoogeSort(SortingAlgorithm):

    def _stoogesort(this, i:int, j:int) -> Generator[SortingAction,None,None]:
        cmp = this.compare(i,j)
        yield (Action.COMPARE,i,j,(i,j))

        if (cmp>0):
            this.swap(i,j)
            yield (Action.SWAP,i,j,(i,j))

        if ((j-i+1)>=3):
            t = (j-i+1)//3

            yield from this._stoogesort(i,j-t)
            yield from this._stoogesort(i+t, j)
            yield from this._stoogesort(i, j-t)

    def make_step(this) -> Generator[SortingAction,None,None]:
        yield from this._stoogesort(0,len(this)-1)

# https://arxiv.org/pdf/1805.04154
class PowerSort(MergeSort):

    MIN_RUN_LENGTH = 16



    @staticmethod
    def _node_power(left:int, right:int, startA:int, startB:int, endB:int):

        def number_of_leading_zeros_32bit(n):
            if n == 0:
                return 32
            leading_zeros = 0
            for i in range(31, -1, -1):
                if (n & (1 << i)) != 0:
                    break
                leading_zeros += 1
            return leading_zeros


        twoN = (right-left + 1) << 1
        l = startA + startB - (left<<1)
        r = startB + endB +1 - (left << 1)
        a =  int((l << 31) // twoN)
        b =  int((r << 31) // twoN)

        xor = a ^ b

        return number_of_leading_zeros_32bit(xor)


    def _extendRunRight(this, m,l):
        while (m<l) and (this.compare(m,m+1)<=0):
            yield (Action.COMPARE,m,m+1,None)
            m+=1

        return m


    def _presortedInsertionSort(this,left,right,nPresorted):
        tmp = [this[x] for x in range(left+nPresorted, right)]

        ins_sort = InsertionSort(tmp)

        for (action, ii, jj, _) in ins_sort.make_step():
            i = this.index(ins_sort[ii])
            j = this.index(ins_sort[jj])

            yield (action, i, j, (left+nPresorted,right))

            if action == Action.SWAP:
                this.swap(i, j)
            elif action == Action.COMPARE:
                this._comparisons += 1

    def _powersort(this, left, right):


        n = right - left + 1
        lgnPlus2 = int(log2(n) + 2)

        leftRunStart = [-1] * lgnPlus2

        top = 0

        startA = left
        endA = yield from this._extendRunRight(startA,right)
        lenA = endA - startA + 1

        if lenA < PowerSort.MIN_RUN_LENGTH:
            endA = min(right,startA + PowerSort.MIN_RUN_LENGTH - 1)
            yield from this._presortedInsertionSort(startA,endA, lenA)

        while (endA < right):
            startB = endA + 1
            endB = yield from this._extendRunRight(startB,right)
            lenB = endB - startB + 1

            if lenB < PowerSort.MIN_RUN_LENGTH:
                endB = min(right, startB + PowerSort.MIN_RUN_LENGTH-1)
                yield from this._presortedInsertionSort(startB,endB, lenB)

            k = PowerSort._node_power(left,right,startA, startB, endB)
            assert k != top

            for l in range(top,k-1,-1):
                if (leftRunStart[l] == -1): continue
                yield from this._inplace_merge(leftRunStart[l],endA)
                startA = leftRunStart[l]
                leftRunStart[l] = -1

            leftRunStart[k] = startA
            top = k
            startA = startB
            endA = endB

        assert endA == right

        for l in range(top,-1,-1):
            if (leftRunStart[l] == -1): continue
            yield from this._inplace_merge(leftRunStart[l], right)





    def make_step(this) -> Generator[SortingAction,None,None]:
        yield from this._powersort(0,len(this)-1)



class BissantiSort(MergeSort):

    def detect_runs(this) -> Generator[SortingAction,None,List[Tuple[int,int]]]:

        n = len(this)
        runs = []

        if (n==0):
            return runs

        start = 0
        cumsum = 0

        for i in range(1,n):
            cmp = this.compare(i-1,i)
            yield (Action.COMPARE,i-1,i,None)

            if ((cmp>0) and (cumsum>n)):
                runs.append((start,i))
                start=i
                cumsum=0
            else:
                cumsum+=this[i]


        runs.append((start,n))

        return runs


    def should_merge(this,runs:List[Tuple[int,int]], idx:int) -> bool:
        calc_power:Callable[[List[Tuple[int,int]]],int] = lambda run : run[1] - run[0]

        n = len(runs)
        if (idx<(n-1)):
            return True if (n==2) else calc_power(runs[idx]) <= calc_power(runs[idx+1])

        return False


    def merge_runs(this,runs:List[Tuple[int,int]]):

        skips = 0

        while (len(runs)>1):
            i=0

            while (i<len(runs)):
                if  (i<(len(runs)-1)) and ((skips>=len(runs)) or this.should_merge(runs,i)):
                    start = runs[i][0]
                    end = runs[i+1][1]

                    yield from this._inplace_merge(start,end-1)

                    #runs.pop(i)
                    runs[i] = (start, end)
                    runs.pop(i+1)

                    i+=2
                    skips=0
                else:
                    i+=1
                    skips+=1



    def make_step(this) -> Generator[SortingAction,None,None]:

        runs = yield from this.detect_runs()


        for a,b in runs:


            tmp = [this[x] for x in range(a,b)]

            ins_sort = InsertionSort(tmp)

            for (action,ii,jj,_) in ins_sort.make_step():
                i = this.index(ins_sort[ii])
                j = this.index(ins_sort[jj])

                yield (action, i, j, None)


                if action == Action.SWAP:
                    this.swap(i,j)
                elif action == Action.COMPARE:
                    this._comparisons+=1

        yield from this.merge_runs(runs)



class CutSort(SortingAlgorithm):

    def make_step(this) -> Generator[SortingAction,None,None]:
        stack = []


        stack.append((0, len(this)-1))

        while (len(stack)>0):
            left, right = stack.pop()

            print (left, right)

            delta = right - left

            if (delta<=0):
                continue
            elif (delta == 1):
                cmp = this.compare(left,right)
                yield (Action.COMPARE,left,right,(left,right))

                if (cmp>0):
                    this.swap(left,right)
                    yield (Action.SWAP, left, right, (left, right))
            elif (delta == 2):
                cursorLeft = left + 1

                cmp = this.compare(left, cursorLeft)
                yield (Action.COMPARE,left,cursorLeft,(left, right))

                if (cmp>0):

                    cmp = this.compare(left, right)
                    yield (Action.COMPARE, left, right, (left, right))

                    if (cmp>0):
                        cmp = this.compare(cursorLeft, right)
                        yield (Action.COMPARE, cursorLeft, right, (left, right))

                        if (cmp>0):
                            this.swap(left, right)
                            yield (Action.SWAP,left, right, (left, right))

                    this.swap(left, cursorLeft)
                    yield (Action.SWAP, left, cursorLeft, (left, right))
                else:

                    cmp = this.compare(cursorLeft, right)
                    yield (Action.COMPARE, cursorLeft, right, (left, right))

                    if (cmp>0):
                        cmp = this.compare(left, right)
                        yield (Action.COMPARE, left, right, (left, right))

                        if (cmp>0):
                            this.swap(left, right)
                            yield (Action.SWAP, left, right, (left, right))

                        this.swap(cursorLeft, right)
                        yield (Action.SWAP, cursorLeft, right, (left, right))

            else:

                cmp = this.compare(left, right)
                yield (Action.COMPARE, left, right, (left, right))

                if (cmp > 0): # if(getValue(left) > getValue(right)) await swap( left, right);
                    this.swap(left, right)
                    yield (Action.SWAP, left, right, (left, right))


                pivotLeft = left
                pivotRight = right
                cursorLeft = left + 1
                cursorRight = right - 1

                doBreak = False

                while True: #do-while

                    while True: # while (getValue(pivotLeft) > getValue(cursorLeft))   cursorLeft++;
                        cmp = this.compare(pivotLeft,cursorLeft)
                        yield (Action.COMPARE, pivotLeft, cursorLeft,(left,right))

                        if (cmp>0): cursorLeft+=1
                        else: break

                    while True: # while (getValue(cursorRight) > getValue(pivotRight)) cursorRight--;
                        cmp = this.compare(cursorRight, pivotRight)
                        yield (Action.COMPARE, cursorRight, pivotRight, (left, right))

                        if (cmp > 0):
                            cursorRight -= 1
                        else:
                            break


                    delta = cursorRight - cursorLeft # switch(cursorRight-cursorLeft)

                    if (delta<0): # case -1:
                        this.swap(pivotLeft, cursorRight) # await swap( pivotLeft, cursorRight);
                        yield (Action.SWAP,pivotLeft,cursorRight,(left,right))

                        this.swap(cursorLeft, pivotRight) # await swap( cursorLeft, pivotRight);
                        yield (Action.SWAP, cursorLeft, pivotRight, (left, right))

                        cursorRight+=1 # cursorRight++;
                        cursorLeft-=1 # cursorLeft--;
                        doBreak = True # doBreak = 1;
                    elif (delta == 0): # case 0:
                        cmp = this.compare(pivotLeft, cursorLeft-1)
                        yield (Action.COMPARE, pivotLeft, cursorLeft-1, (left, right))

                        if (cmp<0): # if (pivotLeft < cursorLeft-1){
                            this.swap(pivotLeft, cursorLeft-1) # await swap( pivotLeft, cursorLeft-1);
                            yield (Action.SWAP, pivotLeft, cursorLeft-1, (left, right))

                        cmp = this.compare(cursorRight+1, pivotRight)
                        yield (Action.COMPARE, cursorRight+1, pivotRight, (left, right))

                        if (cmp < 0): # if (cursorRight + 1< pivotRight){
                            this.swap(cursorRight+1, pivotRight)
                            yield (Action.SWAP,cursorRight+1, pivotRight, (left, right))

                        doBreak = True # doBreak = 1;
                    elif (delta == 1) : #case 1

                        cmp = this.compare(cursorLeft, cursorRight)
                        yield (Action.COMPARE, cursorLeft, cursorRight, (left, right))

                        if (cmp>0): # if (getValue(cursorLeft) > getValue(cursorRight)){
                            this.swap(cursorLeft, cursorRight)
                            yield (Action.SWAP, cursorLeft, cursorRight, (left, right))
                        else:

                            cmp = this.compare(pivotLeft, cursorLeft - 1) #
                            yield (Action.COMPARE, pivotLeft, cursorLeft - 1, (left, right))

                            if (cmp < 0):  # if (pivotLeft < cursorLeft-1){
                                this.swap(pivotLeft, cursorLeft - 1)  # await swap( pivotLeft, cursorLeft-1);
                                yield (Action.SWAP, pivotLeft, cursorLeft - 1, (left, right))

                            cmp = this.compare(cursorRight + 1, pivotRight)
                            yield (Action.COMPARE, cursorRight + 1, pivotRight, (left, right))

                            if (cmp < 0):  # if (cursorRight + 1< pivotRight){
                                this.swap(cursorRight + 1, pivotRight)
                                yield (Action.SWAP, cursorRight + 1, pivotRight, (left, right))

                            doBreak = True


                    else:
                        cmp = this.compare(cursorLeft, cursorRight)
                        yield (Action.COMPARE, cursorLeft, cursorRight, (left,right))

                        if (cmp>0): # (getValue(cursorLeft) > getValue(cursorRight))
                            this.swap(cursorLeft,cursorRight) # await swap(cursorLeft, cursorRight);
                            yield (Action.SWAP, cursorLeft, cursorRight, (left, right))
                        else:
                            cmp = this.compare(pivotLeft, cursorLeft - 1)
                            yield (Action.COMPARE, pivotLeft, cursorLeft - 1, (left, right))

                            if (cmp<0): # # if (pivotLeft < cursorLeft-1){
                                this.swap(pivotLeft, cursorLeft - 1) # await swap( pivotLeft, cursorLeft-1);
                                yield (Action.SWAP, pivotLeft, cursorLeft - 1, (left, right))

                            cmp = this.compare(cursorRight+1, pivotRight)
                            yield (Action.COMPARE, cursorRight+1, pivotRight, (left, right))

                            if (cmp < 0): # if (cursorRight + 1< pivotRight){
                                this.swap(cursorRight+1, pivotRight) # await swap( cursorRight+1, pivotRight);
                                yield (Action.SWAP, cursorRight+1, pivotRight, (left, right))


                            pivotLeft = cursorLeft
                            pivotRight = cursorRight
                            cursorLeft+=1
                            cursorRight-=1

                    if (doBreak): # if (doBreak === 1) break;
                        break

                stack.append((left, cursorLeft))
                stack.append((cursorRight, right))

                cursorLeft-=1
                cursorRight+=1





def get_algorithm(name:str) -> Type[SortingAlgorithm]:

    match (name):
        case "BubbleSort" : return BubbleSort
        case "InsertionSort": return InsertionSort
        case "SelectionSort": return SelectionSort
        case "CombSort": return CombSort
        case "QuickSort": return QuickSort
        case "MergeSort": return MergeSort
        case "HeapSort": return HeapSort
        case "TimSort": return  TimSort
        case "MSDRadixSort": return MSDRadixSort
        case "ICantBelieveItCanSort": return ICantBelieveItCanSort
        case "StoogeSort": return StoogeSort
        case "PowerSort": return PowerSort
        case "CutSort": return CutSort
        case "BissantiSort": return BissantiSort
        case _: NotImplementedError(f"Sorting algorithm {name} not implemented or does not exist.")

