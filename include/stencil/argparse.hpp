#pragma once

#include <string>
#include <vector>

class OptionBase {
public:
  virtual void set_val(const std::string &valStr) = 0;
  virtual const std::string &long_str() = 0;
};

template <typename T> class Option : public OptionBase {
  std::string long_;
  T *val_;

public:
  void set_val(const std::string &valStr) override {}
};

class Flag {
  std::string long_;
  bool *val_;

public:
  Flag(bool &val, const std::string &l) : long_(l), val_(&val) {}

  const std::string &long_str() const noexcept { return long_; }

  void set() const noexcept { *val_ = true; }
};

class PosnlBase {
public:
  virtual bool is_required() = 0;
  virtual PosnlBase *required() = 0;
  virtual void set_val(const std::string &val) = 0;
};

template <typename T> class Positional : public PosnlBase {
  bool required_;
  T *val_;

public:
  Positional(T &val) : required_(false), val_(&val) {}

  PosnlBase *required() override {
    required_ = true;
    return this;
  }

  bool is_required() override { return required_; }

  // use nullpointer type to disambiguate call
  // https://stackoverflow.com/questions/5512910/explicit-specialization-of-template-class-member-function
  void set_val(const std::string &val) { set_val((T *)nullptr, val); }

private:
  // https://stackoverflow.com/questions/5512910/explicit-specialization-of-template-class-member-function
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
  void set_val(int *, const std::string &val) { // convert to int
    *val_ = std::stoi(val);
  }
};

class Parser {

  std::vector<OptionBase *> opts_;
  std::vector<Flag> flags_;
  std::vector<PosnlBase *> posnls_;

  OptionBase *match_opt(const char *arg) const {
    std::string sarg(arg);
    for (int64_t i = int64_t(opts_.size()) - 1; i >= 0; --i) {
      if (opts_[i]->long_str() == sarg) {
        std::cerr << "matched opt " << opts_[i]->long_str() << "\n";
        return opts_[i];
      }
    }
    return nullptr;
  }

  Flag *match_flag(const char *arg) {
    std::string sarg(arg);
    for (int64_t i = int64_t(flags_.size()) - 1; i >= 0; --i) {
      if (flags_[i].long_str() == sarg) {
        return &flags_[i];
      }
    }
    return nullptr;
  }

  static bool starts_with(const std::string &s, const std::string &prefix) {
    if (s.rfind(prefix, 0) == 0) {
      return true;
    }
    return false;
  }

public:
  bool parse(const int argc, const char **argv) {
    size_t pi = 0; // positional argument position
    bool optsOkay = true;
    for (int i = 1; i < argc; ++i) {
      if (argv[i] == std::string("--")) {
        optsOkay = false;
        continue;
      }
      if (optsOkay && starts_with(argv[i], "-")) {
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
        std::cerr << "unrecognized flag " << argv[i] << "\n";
      } else {
        if (pi < posnls_.size()) {
          posnls_[pi]->set_val(argv[i]);
          ++pi;
        } else {
          std::cerr << "encountered unexpected positional argument " << argv[i]
                    << "\n";
        }
      }
    }
    return true;
  };

  template <typename T> void add_option(T &val, const std::string &l) {
    opts_.push_back(new Option<T>(val, l));
  }

  void add_flag(bool &val, const std::string &l) {
    flags_.push_back(Flag(val, l));
  }

  template <typename T> PosnlBase *add_positional(T &val) {
    posnls_.push_back(new Positional<T>(val));
    return posnls_[posnls_.size() - 1];
  }
};
