// Alpacca — small shared utilities. MIT License. See LICENSE.
#include "util.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
#endif

namespace alpacca {

// ---- strings ----------------------------------------------------------

bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (true) {
        size_t next = s.find(sep, pos);
        if (next == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, next - pos));
        pos = next + 1;
    }
    return out;
}

std::string human_size(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double v = (double) bytes;
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        u++;
    }
    char buf[32];
    if (u == 0) {
        snprintf(buf, sizeof(buf), "%llu B", (unsigned long long) bytes);
    } else {
        snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    }
    return buf;
}

std::string now_iso8601() {
    time_t t = time(nullptr);
    struct tm tm_utc;
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
}

// ---- environment / paths ----------------------------------------------

std::string env_or(const char * name, const std::string & def) {
    const char * v = getenv(name);
    return (v && *v) ? std::string(v) : def;
}

static std::string user_home() {
    std::string home = env_or("HOME", "");
#ifdef _WIN32
    if (home.empty()) {
        home = env_or("USERPROFILE", "");
    }
#endif
    return home;
}

fs::path alpacca_home() {
    std::string explicit_home = env_or("ALPACCA_HOME", "");
    if (!explicit_home.empty()) {
        return fs::path(explicit_home);
    }
    std::string home = user_home();
    if (home.empty()) {
        fail("cannot determine home directory (set $ALPACCA_HOME)");
    }
    return fs::path(home) / ".alpacca";
}

fs::path self_exe_dir() {
#ifdef _WIN32
    wchar_t buf[32768];
    DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof(buf) / sizeof(buf[0])));
    if (n > 0) {
        return fs::path(std::wstring(buf, n)).parent_path();
    }
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return fs::path(buf).parent_path();
    }
#endif
    return fs::current_path();
}

std::string sanitize_component(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        bool ok = isalnum((unsigned char) c) || c == '.' || c == '-' || c == '_' || c == '+';
        out.push_back(ok ? c : '_');
    }
    if (out.empty() || out == "." || out == "..") {
        out = "_";
    }
    return out;
}

// ---- processes ---------------------------------------------------------

#ifdef _WIN32

static std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), &w[0], n);
    return w;
}

// Quote one argument following the MSVC runtime's parsing rules.
static std::string quote_win_arg(const std::string & a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) {
        return a;
    }
    std::string r = "\"";
    size_t backslashes = 0;
    for (char c : a) {
        if (c == '\\') {
            backslashes++;
            r.push_back(c);
        } else if (c == '"') {
            r.append(backslashes + 1, '\\');
            r.push_back('"');
            backslashes = 0;
        } else {
            backslashes = 0;
            r.push_back(c);
        }
    }
    r.append(backslashes, '\\');
    r.push_back('"');
    return r;
}

int run_process(const std::vector<std::string> & args,
                const std::string * stdin_data,
                std::string * out) {
    if (args.empty()) {
        return -1;
    }
    std::string cmdline;
    for (size_t i = 0; i < args.size(); i++) {
        if (i) {
            cmdline.push_back(' ');
        }
        cmdline += quote_win_arg(args[i]);
    }
    std::wstring wcmd = widen(cmdline);

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr;
    if (stdin_data) {
        if (!CreatePipe(&in_r, &in_w, &sa, 0)) {
            return -1;
        }
        SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    }
    if (out) {
        if (!CreatePipe(&out_r, &out_w, &sa, 0)) {
            return -1;
        }
        SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    }

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = stdin_data ? in_r : GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = out ? out_w : GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(nullptr, &wcmd[0], nullptr, nullptr, TRUE, 0,
                             nullptr, nullptr, &si, &pi);
    if (stdin_data) {
        CloseHandle(in_r);
    }
    if (out) {
        CloseHandle(out_w);
    }
    if (!ok) {
        if (stdin_data) {
            CloseHandle(in_w);
        }
        if (out) {
            CloseHandle(out_r);
        }
        fprintf(stderr, "alpacca: failed to run '%s'\n", args[0].c_str());
        return -1;
    }

    if (stdin_data) {
        DWORD written = 0;
        size_t off = 0;
        while (off < stdin_data->size() &&
               WriteFile(in_w, stdin_data->data() + off,
                         (DWORD) (stdin_data->size() - off), &written, nullptr) &&
               written > 0) {
            off += written;
        }
        CloseHandle(in_w);
    }
    if (out) {
        char buf[8192];
        DWORD n = 0;
        while (ReadFile(out_r, buf, sizeof(buf), &n, nullptr) && n > 0) {
            out->append(buf, n);
        }
        CloseHandle(out_r);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int) code;
}

void exec_replace(const std::vector<std::string> & args) {
    // No execvp on Windows: run the child in this console, ignore Ctrl+C in
    // the parent (the child owns it), and exit with the child's code.
    SetConsoleCtrlHandler(nullptr, TRUE);
    int rc = run_process(args, nullptr, nullptr);
    exit(rc < 0 ? 127 : rc);
}

bool on_path(const std::string & name) {
    std::wstring wname = widen(name);
    wchar_t found[32768];
    return SearchPathW(nullptr, wname.c_str(), L".exe", 32768, found, nullptr) > 0 ||
           SearchPathW(nullptr, wname.c_str(), nullptr, 32768, found, nullptr) > 0;
}

#else // POSIX

int run_process(const std::vector<std::string> & args,
                const std::string * stdin_data,
                std::string * out) {
    if (args.empty()) {
        return -1;
    }

    int in_pipe[2]  = { -1, -1 };
    int out_pipe[2] = { -1, -1 };
    if (stdin_data && pipe(in_pipe) != 0) {
        return -1;
    }
    if (out && pipe(out_pipe) != 0) {
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }

    if (pid == 0) { // child
        if (stdin_data) {
            dup2(in_pipe[0], STDIN_FILENO);
            close(in_pipe[0]);
            close(in_pipe[1]);
        }
        if (out) {
            dup2(out_pipe[1], STDOUT_FILENO);
            close(out_pipe[0]);
            close(out_pipe[1]);
        }
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto & a : args) {
            argv.push_back(const_cast<char *>(a.c_str()));
        }
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        fprintf(stderr, "alpacca: failed to run '%s': %s\n", argv[0], strerror(errno));
        _exit(127);
    }

    // parent
    if (stdin_data) {
        close(in_pipe[0]);
        size_t off = 0;
        while (off < stdin_data->size()) {
            ssize_t w = write(in_pipe[1], stdin_data->data() + off, stdin_data->size() - off);
            if (w <= 0) {
                break;
            }
            off += (size_t) w;
        }
        close(in_pipe[1]);
    }
    if (out) {
        close(out_pipe[1]);
        char buf[8192];
        ssize_t r;
        while ((r = read(out_pipe[0], buf, sizeof(buf))) > 0) {
            out->append(buf, (size_t) r);
        }
        close(out_pipe[0]);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return -1;
}

void exec_replace(const std::vector<std::string> & args) {
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    fprintf(stderr, "alpacca: failed to exec '%s': %s\n", argv[0], strerror(errno));
}

bool on_path(const std::string & name) {
    std::string path = env_or("PATH", "");
    for (const auto & dir : split(path, ':')) {
        if (dir.empty()) {
            continue;
        }
        fs::path candidate = fs::path(dir) / name;
        std::error_code ec;
        if (fs::exists(candidate, ec) && access(candidate.c_str(), X_OK) == 0) {
            return true;
        }
    }
    return false;
}

#endif // _WIN32

// ---- errors ------------------------------------------------------------

void fail(const std::string & msg) {
    throw std::runtime_error(msg);
}

} // namespace alpacca
