#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace argparse {

namespace detail {
template <typename T> const char *type_str();
template <> const char *type_str<int32_t>() { return "INT32"; }
template <> const char *type_str<int64_t>() { return "INT64"; }
template <> const char *type_str<std::string>() { return "STR"; }
template <> const char *type_str<float>() { return "F32"; }
template <> const char *type_str<double>() { return "F64"; }
template <> const char *type_str<size_t>() { return "SIZE_T"; }
} // namespace detail

class Parser;

/* Interface for an object representing a command-line option
 */
class OptionBase {
  friend class Parser;

public:
  /* return the short and long options
   */
  virtual const std::string &short_str() = 0;
  virtual const std::string &long_str() = 0;

  /* return the help string
   */
  virtual const std::string &help_str() = 0;

  /* set help string
   */
  virtual void help(const std::string &s) = 0;

private:
  /* a string representing the type of the option (INT, STR, FLOAT)
   */
  virtual const char *type_str() = 0;

  /* set the value provided to this option
   */
  virtual void set_val(const std::string &valStr) = 0;
};

template <typename T> class Option : public OptionBase {
  std::string short_;
  std::string long_;
  std::string help_;
  T *val_;

public:
  Option(T &val, const std::string &l, const std::string &s = "")
      : long_(l), short_(s), val_(&val) {}
  const std::string &long_str() override { return long_; }
  const std::string &short_str() override { return short_; }
  const std::string &help_str() override { return help_; }

  void help(const std::string &s) { help_ = s; }

private:
  /* uses a T* = nullpointer to disambiguate the call
   */
  const char *type_str() override { return detail::type_str<T>(); }
  void set_val(const std::string &val) override { set_val((T *)nullptr, val); }

  /* how to parse supported types
   */
  void set_val(size_t *, const std::string &val) { // convert to size_t
    *val_ = std::stoull(val);
  }
  void set_val(double *, const std::string &val) { // convert to double
    *val_ = std::stod(val);
  }
  void set_val(float *, const std::string &val) { // convert to float
    *val_ = std::stof(val);
  }
  void set_val(int32_t *, const std::string &val) { // convert to int32_t
    *val_ = std::stoi(val);
  }
  void set_val(int64_t *, const std::string &val) { // convert to int64_t
    *val_ = std::stoi(val);
  }
  void set_val(std::string *, const std::string &val) { // convert to string
    *val_ = val;
  }
};

/* Represents a command-line flag
 */
class Flag {
  std::string long_;
  std::string short_;
  std::string help_;
  bool *val_;

public:
  Flag(bool &val, const std::string &l, const std::string &s)
      : long_(l), short_(s), val_(&val) {}

  const std::string &long_str() const noexcept { return long_; }
  const std::string &short_str() const noexcept { return short_; }

  void set() const noexcept { *val_ = true; }

  void help(const std::string &s) { help_ = s; }

  const std::string &help_str() const noexcept { return help_; }
};

class PosnlBase {
  friend class Parser;

public:
  virtual bool is_required() = 0;
  virtual PosnlBase *required() = 0;
  virtual void set_val(const std::string &val) = 0;
  virtual bool found() = 0;

private:
  virtual const char *type_str() = 0;
};

template <typename T> class Positional : public PosnlBase {
  bool required_;
  T *val_;
  bool found_;

public:
  Positional(T &val) : required_(false), val_(&val), found_(false) {}

  PosnlBase *required() override {
    required_ = true;
    return this;
  }

  bool is_required() override { return required_; }

  // use nullpointer type to disambiguate call
  // https://stackoverflow.com/questions/5512910/explicit-specialization-of-template-class-member-function
  void set_val(const std::string &val) {
    found_ = true;
    set_val((T *)nullptr, val);
  }

  bool found() override { return found_; }

private:
  const char *type_str() override { return detail::type_str<T>(); }

  template <typename C>
  void get_as(C *, const std::string &val) { // to be overridden
  }
  void set_val(size_t *, const std::string &val) { // convert to size_t
    *val_ = std::stoull(val);
  }
  void set_val(double *, const std::string &val) { // convert to double
    *val_ = std::stod(val);
  }
  void set_val(float *, const std::string &val) { // convert to float
    *val_ = std::stof(val);
  }
  void set_val(int32_t *, const std::string &val) { // convert to int32_t
    *val_ = std::stoi(val);
  }
  void set_val(int64_t *, const std::string &val) { // convert to int64_t
    *val_ = std::stoi(val);
  }
  void set_val(std::string *, const std::string &val) { // convert to string
    *val_ = val;
  }
};

class Parser {

  std::string description_;
  bool noUnrecognized_; // error on unrecognized flags / opts
  bool help_;           // help has been requested
  bool consume_;        // remove consumed values from argc, argv

  std::vector<OptionBase *> opts_;
  std::vector<Flag> flags_;
  std::vector<PosnlBase *> posnls_;

  std::string argv0_; // will hold the name of the most recent parse(...)

  static bool starts_with(const std::string &s, const std::string &prefix) {
    if (s.rfind(prefix, 0) == 0) {
      return true;
    }
    return false;
  }

  OptionBase *match_opt(const char *arg) const {
    std::string sarg(arg);

    // match options back-to-front to prefer later duplicates
    for (int64_t i = int64_t(opts_.size()) - 1; i >= 0; --i) {
      if (opts_[i]->long_str() == sarg || opts_[i]->short_str() == sarg) {
        return opts_[i];
      }
    }
    return nullptr;
  }

  Flag *match_flag(const char *arg) {
    std::string sarg(arg);

    // match flags back-to-front to prefer later duplicates
    for (int64_t i = int64_t(flags_.size()) - 1; i >= 0; --i) {
      if (flags_[i].long_str() == sarg || flags_[i].short_str() == sarg) {
        return &flags_[i];
      }
    }
    return nullptr;
  }

public:
  Parser()
      : description_("an argparse-powered program"), noUnrecognized_(false),
        help_(false), consume_(true) {
    add_flag(help_, "--help", "-h")->help("Print help message");
  }
  Parser(const std::string &description)
      : description_(description), noUnrecognized_(false), help_(false),
        consume_(true) {
    add_flag(help_, "--help", "-h")->help("Print help message");
  }

  bool parse(int &argc, char **argv) {

    std::vector<char *> newArgv;
    if (argc > 0) {
      newArgv.push_back(argv[0]);

      if (argv[0]) {
        argv0_ = argv[0];
      } else {
        argv0_ = "";
      }
    }

    size_t pi = 0;        // positional argument position
    bool optsOkay = true; // okay to interpret as opt/flag
    for (int i = 1; i < argc; ++i) {

      // try interpreting as a flag or option if it looks like one
      if (optsOkay && starts_with(argv[i], "-")) {
        // '--' indicates only positional arguments follow
        // the second '--' should be interpreted as a positional argument
        if (argv[i] == std::string("--")) {
          optsOkay = false;
          continue;
        }
        OptionBase *opt = match_opt(argv[i]);
        if (opt) {
          opt->set_val(argv[i + 1]);
          ++i;
          continue;
        }
        Flag *flag = match_flag(argv[i]);
        if (flag) {
          flag->set();
          continue;
        }
        newArgv.push_back(argv[i]);
        if (noUnrecognized_) {
          std::cerr << "unrecognized " << argv[i] << "\n";
          return false;
        }
      } else { // otherwise try it as positional
        if (pi < posnls_.size()) {
          posnls_[pi]->set_val(argv[i]);
          ++pi;
        } else {
          newArgv.push_back(argv[i]);
          std::cerr << "encountered unexpected positional argument " << pi
                    << ": " << argv[i] << "\n";
        }
      }
    }

    for (; pi < posnls_.size(); ++pi) {
      if (posnls_[pi]->is_required()) {
        std::cerr << "missing required positional argument " << pi << "\n";
        return false;
      }
    }

    if (consume_) {
      argc = newArgv.size();
      for (int i = 0; i < argc; ++i) {
        argv[i] = newArgv[i];
      }
    }

    return true;
  };

  template <typename T>
  OptionBase *add_option(T &val, const std::string &l,
                         const std::string &s = "") {
    opts_.push_back(new Option<T>(val, l, s));
    return opts_.back();
  }

  Flag *add_flag(bool &val, const std::string &l, const std::string &s = "") {
    flags_.push_back(Flag(val, l, s));
    return &(flags_.back());
  }

  template <typename T> PosnlBase *add_positional(T &val) {
    posnls_.push_back(new Positional<T>(val));
    return posnls_.back();
  }

  std::string help() const {
    std::stringstream ss;

    ss << "Usage: " << argv0_ << " [OPTION]...";

    for (auto &p : posnls_) {
      ss << " ";
      ss << (p->is_required() ? "" : "[");
      ss << p->type_str();
      ss << (p->is_required() ? "" : "]");
    }
    ss << "\n";

    ss << description_ << "\n";

    for (auto &o : opts_) {
      ss << "  ";
      if (!o->short_str().empty()) {
        ss << o->short_str() << ", ";
      }
      ss << o->long_str() << "\t" << o->type_str();
      ss << "\t\t" << o->help_str();
      ss << "\n";
    }

    for (auto &f : flags_) {
      ss << "  " << f.short_str() << ", " << f.long_str();
      ss << "\t\t" << f.help_str();
      ss << "\n";
    }

    return ss.str();
  }

  /*! \brief error on unrecognized flags and options
   */
  void no_unrecognized() { noUnrecognized_ = true; }

  /*! \brief don't modify argc/argv
   */
  void no_consume() { consume_ = false; }

  bool need_help() const noexcept { return help_; }
};

} // namespace argparse