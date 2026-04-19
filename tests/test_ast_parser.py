"""Tests for tree-sitter AST parsing module."""

import pytest

from cairn.ast_parser import (
    extract_file_ast,
    lang_for_file,
    parse_file,
    _check_available,
)


pytestmark = pytest.mark.skipif(
    not _check_available(),
    reason="tree-sitter not installed",
)


class TestLangForFile:
    def test_python(self):
        assert lang_for_file("foo.py") == "python"

    def test_typescript(self):
        assert lang_for_file("app.ts") == "typescript"

    def test_tsx(self):
        assert lang_for_file("Component.tsx") == "tsx"

    def test_javascript(self):
        assert lang_for_file("index.js") == "javascript"
        assert lang_for_file("lib.mjs") == "javascript"

    def test_go(self):
        assert lang_for_file("main.go") == "go"

    def test_rust(self):
        assert lang_for_file("lib.rs") == "rust"

    def test_c(self):
        assert lang_for_file("util.c") == "c"
        assert lang_for_file("util.h") == "c"

    def test_cpp(self):
        assert lang_for_file("main.cpp") == "cpp"
        assert lang_for_file("main.cc") == "cpp"
        assert lang_for_file("main.hpp") == "cpp"

    def test_unknown(self):
        assert lang_for_file("data.json") is None
        assert lang_for_file("Makefile") is None


class TestPythonExtraction:
    def test_function(self):
        result = extract_file_ast("test.py", b"def hello(name: str) -> str:\n    return name\n")
        assert result["language"] == "python"
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["name"] == "hello"
        assert "name: str" in fns[0]["params"]
        assert fns[0]["kind"] == "function"
        assert fns[0]["line"] == 1

    def test_class_with_methods(self):
        code = b"""
class Dog:
    def bark(self):
        return "woof"
    def fetch(self, item: str) -> str:
        return item
"""
        result = extract_file_ast("test.py", code)
        classes = result["classes"]
        assert len(classes) == 1
        assert classes[0]["name"] == "Dog"
        methods = classes[0]["methods"]
        assert len(methods) == 2
        assert methods[0]["name"] == "Dog.bark"
        assert methods[0]["kind"] == "method"

    def test_decorated_function(self):
        code = b"""
@app.route("/hello")
def hello():
    pass
"""
        result = extract_file_ast("test.py", code)
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["name"] == "hello"

    def test_imports(self):
        code = b"""
import os
from pathlib import Path
from typing import Optional, List
import json
"""
        result = extract_file_ast("test.py", code)
        imports = result["imports"]
        assert len(imports) >= 3
        modules = [i["module"] for i in imports]
        assert "os" in modules
        assert "pathlib" in modules

    def test_class_inheritance(self):
        code = b"""
class Animal:
    pass

class Dog(Animal):
    def bark(self):
        pass
"""
        result = extract_file_ast("test.py", code)
        classes = result["classes"]
        assert len(classes) == 2
        dog = [c for c in classes if c["name"] == "Dog"][0]
        assert "Animal" in dog["bases"]

    def test_empty_file(self):
        result = extract_file_ast("test.py", b"")
        assert result["functions"] == []
        assert result["classes"] == []
        assert result["imports"] == []


class TestJavaScriptExtraction:
    def test_function_declaration(self):
        code = b"function greet(name) { return name; }\n"
        result = extract_file_ast("test.js", code)
        assert result["language"] == "javascript"
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["name"] == "greet"

    def test_class(self):
        code = b"""
class Animal {
    constructor(name) {
        this.name = name;
    }
    speak() {
        return this.name;
    }
}
"""
        result = extract_file_ast("test.js", code)
        classes = result["classes"]
        assert len(classes) == 1
        assert classes[0]["name"] == "Animal"
        assert len(classes[0]["methods"]) >= 1

    def test_arrow_function(self):
        code = b"const add = (a, b) => a + b;\n"
        result = extract_file_ast("test.js", code)
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["name"] == "add"

    def test_imports(self):
        code = b"""
import { readFile } from 'fs';
import path from 'path';
import * as util from './util';
"""
        result = extract_file_ast("test.js", code)
        imports = result["imports"]
        assert len(imports) == 3
        modules = [i["module"] for i in imports]
        assert "fs" in modules
        assert "path" in modules

    def test_exports(self):
        code = b"""
export function hello() { return "hi"; }
export class Foo {}
"""
        result = extract_file_ast("test.js", code)
        exports = result["exports"]
        assert len(exports) >= 1

    def test_async_function(self):
        code = b"const fetchData = async (url) => { return url; };\n"
        result = extract_file_ast("test.js", code)
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["async"] is True


class TestTypeScriptExtraction:
    def test_ts_function(self):
        code = b"function greet(name: string): string { return name; }\n"
        result = extract_file_ast("test.ts", code)
        assert result["language"] == "typescript"
        assert len(result["functions"]) == 1

    def test_tsx_component(self):
        code = b"""
import React from 'react';
function App() { return null; }
export default App;
"""
        result = extract_file_ast("test.tsx", code)
        assert result["language"] == "tsx"
        assert len(result["functions"]) >= 1


class TestGoExtraction:
    def test_function(self):
        code = b"""package main

func main() {
    fmt.Println("hello")
}

func add(a int, b int) int {
    return a + b
}
"""
        result = extract_file_ast("test.go", code)
        assert result["language"] == "go"
        fns = result["functions"]
        assert len(fns) == 2
        names = [f["name"] for f in fns]
        assert "main" in names
        assert "add" in names

    def test_struct_and_method(self):
        code = b"""package main

type Dog struct {
    Name string
    Age  int
}

func (d *Dog) Bark() string {
    return "woof"
}
"""
        result = extract_file_ast("test.go", code)
        types = result["types"]
        assert len(types) == 1
        assert types[0]["name"] == "Dog"
        assert types[0]["kind"] == "struct"
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["name"] == "Bark"
        assert fns[0]["kind"] == "method"

    def test_imports(self):
        code = b"""package main

import (
    "fmt"
    "os"
)
"""
        result = extract_file_ast("test.go", code)
        imports = result["imports"]
        assert len(imports) == 2
        modules = [i["module"] for i in imports]
        assert "fmt" in modules
        assert "os" in modules

    def test_interface(self):
        code = b"""package main

type Reader interface {
    Read(p []byte) (n int, err error)
}
"""
        result = extract_file_ast("test.go", code)
        types = result["types"]
        assert len(types) == 1
        assert types[0]["name"] == "Reader"
        assert types[0]["kind"] == "interface"


class TestRustExtraction:
    def test_function(self):
        code = b"""
pub fn hello(name: &str) -> String {
    format!("Hello {}", name)
}

fn private_fn() {}
"""
        result = extract_file_ast("test.rs", code)
        assert result["language"] == "rust"
        fns = result["functions"]
        assert len(fns) == 2
        hello = [f for f in fns if f["name"] == "hello"][0]
        assert hello["public"] is True

    def test_struct_and_impl(self):
        code = b"""
pub struct Dog {
    name: String,
    age: u32,
}

impl Dog {
    pub fn new(name: String) -> Self {
        Dog { name, age: 0 }
    }
    fn bark(&self) -> &str {
        "woof"
    }
}
"""
        result = extract_file_ast("test.rs", code)
        structs = result["structs"]
        assert len(structs) == 1
        assert structs[0]["name"] == "Dog"
        impls = result["impls"]
        assert len(impls) == 1
        assert impls[0]["name"] == "Dog"
        assert len(impls[0]["methods"]) == 2

    def test_trait_impl(self):
        code = b"""
trait Speak {
    fn speak(&self) -> &str;
}

struct Cat;

impl Speak for Cat {
    fn speak(&self) -> &str {
        "meow"
    }
}
"""
        result = extract_file_ast("test.rs", code)
        traits = result["traits"]
        assert len(traits) == 1
        assert traits[0]["name"] == "Speak"
        impls = result["impls"]
        assert len(impls) == 1
        assert impls[0]["trait"] is not None

    def test_enum(self):
        code = b"enum Color { Red, Green, Blue }\n"
        result = extract_file_ast("test.rs", code)
        enums = result["enums"]
        assert len(enums) == 1
        assert enums[0]["name"] == "Color"

    def test_use_declarations(self):
        code = b"""
use std::io;
use std::collections::HashMap;
"""
        result = extract_file_ast("test.rs", code)
        imports = result["imports"]
        assert len(imports) == 2


class TestCExtraction:
    def test_function(self):
        code = b"""
int add(int a, int b) {
    return a + b;
}
"""
        result = extract_file_ast("test.c", code)
        assert result["language"] == "c"
        fns = result["functions"]
        assert len(fns) == 1
        assert fns[0]["name"] == "add"

    def test_struct(self):
        code = b"""
struct Point {
    int x;
    int y;
};
"""
        result = extract_file_ast("test.c", code)
        structs = result["structs"]
        assert len(structs) == 1
        assert structs[0]["name"] == "Point"
        assert len(structs[0]["fields"]) == 2

    def test_includes(self):
        code = b"""
#include <stdio.h>
#include "myheader.h"
"""
        result = extract_file_ast("test.c", code)
        includes = result["includes"]
        assert len(includes) == 2


class TestCppExtraction:
    def test_class(self):
        code = b"""
class Animal {
public:
    void speak() {}
    int getAge() { return 0; }
};
"""
        result = extract_file_ast("test.cpp", code)
        assert result["language"] == "cpp"
        classes = result["classes"]
        assert len(classes) == 1
        assert classes[0]["name"] == "Animal"

    def test_namespace(self):
        code = b"""
namespace mylib {
    int foo() { return 42; }
}
"""
        result = extract_file_ast("test.cpp", code)
        namespaces = result["namespaces"]
        assert len(namespaces) == 1
        assert namespaces[0]["name"] == "mylib"


class TestEdgeCases:
    def test_unsupported_extension(self):
        result = extract_file_ast("data.json", b'{"key": "value"}')
        assert result is None

    def test_syntax_error_still_parses(self):
        code = b"def broken(\n    # no closing paren"
        result = extract_file_ast("test.py", code)
        assert result is not None

    def test_binary_content(self):
        result = extract_file_ast("test.py", b"\x00\x01\x02\xff")
        assert result is not None
