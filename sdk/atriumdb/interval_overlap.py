import timeit
import random


def is_interval_overlap(intervals):
    """
    is_interval_overlap(intervals)

    Return True if any of the tuples in the list overlap with eachother.

    Parameters
    ----------
    intervals : list of tuples of integers
        The tuples represent intervals like [3, 5), meaning starting at 3 and up to but not including 5.

    Returns
    -------
    boolean
        True if any of the tuples in the list overlap with eachother.

    Raises
    ------
    ValueError
        If the input list is length 0.

    Examples
    --------
    >>>is_interval_overlap([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13)])
    False
    >>>is_interval_overlap([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13), (19, 22)])
    True
    """
    if len(intervals) == 0:
        raise ValueError("Input list is length 0")
    for i in range(len(intervals)):
        for j in range(i+1, len(intervals)):
            if intervals[i][1] > intervals[j][0] and intervals[i][0] < intervals[j][1]:
                return True
    return False


def is_interval_overlap_2(intervals):
    """
        is_interval_overlap_2(intervals)

        Return True if any of the tuples in the list overlap with eachother.

        Parameters
        ----------
        intervals : list of tuples of integers
            The tuples represent intervals like [3, 5), meaning starting at 3 and up to but not including 5.

        Returns
        -------
        boolean
            True if any of the tuples in the list overlap with eachother.

        Raises
        ------
        ValueError
            If the input list is length 0.

        Examples
        --------
        >>>is_interval_overlap([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13)])
        False
        >>>is_interval_overlap([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13), (19, 22)])
        True
        """
    if len(intervals) == 0:
        raise ValueError("Input list is length 0")
    intervals.sort()
    for i in range(len(intervals)-1):
        if intervals[i][1] > intervals[i+1][0]:
            return True
    return False


def generate_n_intervals(n, search_space_factor=1_000_000, interval_size_factor=1_000_000):
    result = []
    for i in range(n):
        first_number = random.randint(0, 1000 * search_space_factor)

        second_number = first_number + random.randint(1, max(2, n // interval_size_factor))
        result.append((first_number, second_number))
    return result


def time_is_interval_overlap_with_generate_n_intervals(number=100):
    for n in [10, 100, 1000, 10000]:
        print()
        print("n = {}".format(n))
        print("is_interval_overlap")
        print(timeit.timeit("is_interval_overlap(generate_n_intervals({}))"
                            .format(n), setup="from __main__ import is_interval_overlap, generate_n_intervals", number=number))
        print("is_interval_overlap_2")
        print(timeit.timeit("is_interval_overlap_2(generate_n_intervals({}))"
                            .format(n), setup="from __main__ import is_interval_overlap_2, generate_n_intervals", number=number))


if __name__ == "__main__":
    time_is_interval_overlap_with_generate_n_intervals(number=10)
