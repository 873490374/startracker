from program.tracker.kvector_calculator import KVectorCalculator


class TestKVector:

    def test_k_vector(self):
        y_vector = [
            0.7, 0.9, 0.51, 0.123, 0.62, 0.562, 0.746, 0.32, 0.86, 0.561]
        y_a = 0.32
        y_b = 0.51
        kvector = KVectorCalculator()
        db_vector = kvector.make_kvector(y_vector)

        assert db_vector.k_vector == [0, 0, 0, 1, 1, 4, 5, 7, 8, 10]
        assert db_vector.s_vector == [
            0.123, 0.32, 0.51, 0.561, 0.562, 0.62, 0.7, 0.746, 0.86, 0.9]
        assert kvector.find_in_kvector(y_a, y_b, db_vector) == [
            0.32, 0.51, 0.561, 0.562]

    def test_k_vector_catalog(self):
        pass
