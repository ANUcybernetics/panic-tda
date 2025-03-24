import ray


def test_ray_basic_functionality():
    """Test that Ray can be initialized and used for simple parallel computation."""

    # Define a simple remote function
    @ray.remote
    def square(x):
        return x * x

    # Test parallel execution
    futures = [square.remote(i) for i in range(5)]
    results = ray.get(futures)

    # Verify results
    assert results == [0, 1, 4, 9, 16]

    # Clean up Ray
    ray.shutdown()


def test_ray_dynamic_generator():
    """Test that a Ray dynamic generator returns correct results and terminates properly."""

    # Define a generator function with dynamic returns
    @ray.remote(num_returns="dynamic")
    def yield_values(n):
        for i in range(n):
            yield i * 10
        # Generator should stop after yielding n values

    # Get the dynamic generator reference
    dynamic_ref = yield_values.remote(5)

    # Get the generator object
    ref_generator = ray.get(dynamic_ref)

    # Collect all values from the generator
    results = []
    for ref in ref_generator:
        results.append(ray.get(ref))

    # Verify all expected values were yielded
    assert results == [0, 10, 20, 30, 40]

    # Verify generator is exhausted (no more values)
    assert len(list(ref_generator)) == 0

    ray.shutdown()
