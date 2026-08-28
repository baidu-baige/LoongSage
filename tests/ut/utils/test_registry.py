"""Unit tests for coda/utils/registry.py."""

import unittest

from coda.utils.registry import Registry


class TestRegistryRegister(unittest.TestCase):

    def setUp(self):
        self.reg = Registry("test")

    def test_register_and_get_roundtrip(self):
        @self.reg.register("foo")
        def foo():
            pass
        self.assertIs(self.reg.get("foo"), foo)

    def test_register_returns_original_function_unchanged(self):
        def bar():
            return 42
        result = self.reg.register("bar")(bar)
        self.assertIs(result, bar)
        self.assertEqual(result(), 42)

    def test_register_non_callable(self):
        """Registry is not restricted to callables."""
        self.reg.register("val")(123)
        self.assertEqual(self.reg.get("val"), 123)

    def test_duplicate_name_raises_value_error(self):
        self.reg.register("dup")(lambda: None)
        with self.assertRaises(ValueError):
            self.reg.register("dup")(lambda: None)

    def test_error_message_contains_registry_name_and_key(self):
        self.reg.register("x")(lambda: None)
        with self.assertRaises(ValueError) as ctx:
            self.reg.register("x")(lambda: None)
        msg = str(ctx.exception)
        self.assertIn("test", msg)
        self.assertIn("x", msg)


class TestRegistryGet(unittest.TestCase):

    def setUp(self):
        self.reg = Registry("test")

    def test_get_unknown_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.reg.get("nonexistent")

    def test_error_message_lists_available_keys(self):
        self.reg.register("a")(lambda: None)
        with self.assertRaises(KeyError) as ctx:
            self.reg.get("missing")
        self.assertIn("a", str(ctx.exception))

    def test_get_returns_correct_object(self):
        sentinel = object()
        self.reg.register("s")(sentinel)
        self.assertIs(self.reg.get("s"), sentinel)


class TestRegistryIntrospection(unittest.TestCase):

    def setUp(self):
        self.reg = Registry("test")

    def test_len_empty(self):
        self.assertEqual(len(self.reg), 0)

    def test_len_after_register(self):
        self.reg.register("a")(lambda: None)
        self.reg.register("b")(lambda: None)
        self.assertEqual(len(self.reg), 2)

    def test_contains_registered_key(self):
        self.reg.register("x")(lambda: None)
        self.assertIn("x", self.reg)

    def test_not_contains_unknown_key(self):
        self.assertNotIn("y", self.reg)

    def test_keys_returns_all_names(self):
        self.reg.register("p")(lambda: None)
        self.reg.register("q")(lambda: None)
        self.assertCountEqual(self.reg.keys(), ["p", "q"])

    def test_repr_contains_registry_name(self):
        reg = Registry("my_registry")
        self.assertIn("my_registry", repr(reg))


class TestRegistryIsolation(unittest.TestCase):

    def test_two_registries_are_independent(self):
        r1 = Registry("r1")
        r2 = Registry("r2")
        r1.register("shared")(lambda: "r1")
        r2.register("shared")(lambda: "r2")
        self.assertEqual(r1.get("shared")(), "r1")
        self.assertEqual(r2.get("shared")(), "r2")

    def test_register_in_one_does_not_appear_in_other(self):
        r1 = Registry("r1")
        r2 = Registry("r2")
        r1.register("only_in_r1")(lambda: None)
        self.assertNotIn("only_in_r1", r2)


if __name__ == "__main__":
    unittest.main()
