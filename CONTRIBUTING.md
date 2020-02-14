# Guidelines for Contributors

* use `inline` for functions in header files (not static)
    * `static` says that each TU should provide its own function
    * `inline` says to supress the one-definition rule, and allows the compilers to merge definitions later.
* use signed integer types wherever possible