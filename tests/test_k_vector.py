from program.tracker.kvector import KVector


def k_vector_wrapper():
    y_vector = [0.7, 0.9, 0.51, 0.123, 0.62, 0.562, 0.746, 0.32, 0.86, 0.561]

    kvector = KVector()
    kvector.make_kvector(y_vector)
    return kvector


class TestKVector:
    #
    # def test_k_vector(self, benchmark):
    #
    #     kvector = benchmark(k_vector_wrapper)
    #     assert kvector.k_vector == [0, 0, 0, 1, 1, 4, 5, 7, 8, 10]
    #     assert kvector.s_vector == [
    #         0.123, 0.32, 0.51, 0.561, 0.562, 0.62, 0.7, 0.746, 0.86, 0.9]
    #     assert kvector.find_in_kvector(0.32, 0.51) == [
    #         0.32, 0.51, 0.561, 0.562]

    pass
