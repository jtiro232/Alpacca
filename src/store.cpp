// Alpacca — local model store. MIT License. See LICENSE.
#include "store.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

namespace alpacca {

// ---- references ---------------------------------------------------------

static bool looks_like_path(const std::string & s) {
    if (starts_with(s, "/") || starts_with(s, "./") || starts_with(s, "../") || starts_with(s, "~/")) {
        return true;
    }
#ifdef _WIN32
    if (s.size() >= 2 && isalpha((unsigned char) s[0]) && s[1] == ':') { // C:\...
        return true;
    }
    if (starts_with(s, ".\\") || starts_with(s, "..\\") || starts_with(s, "\\\\")) {
        return true;
    }
#endif
    if (ends_with(to_lower(s), ".gguf")) {
        std::error_code ec;
        if (fs::exists(s, ec)) {
            return true;
        }
    }
    return false;
}

static fs::path expand_tilde(const std::string & s) {
    if (starts_with(s, "~/")) {
        return fs::path(env_or("HOME", env_or("USERPROFILE", ""))) / s.substr(2);
    }
    return fs::path(s);
}

// Split "name[:tag]" — the tag is whatever follows the last ':' as long as it
// contains no '/', so "hf:org/repo" keeps its prefix handling elsewhere.
static void split_tag(const std::string & in, std::string & name, std::string & tag) {
    size_t colon = in.rfind(':');
    if (colon != std::string::npos && in.find('/', colon) == std::string::npos) {
        name = in.substr(0, colon);
        tag = in.substr(colon + 1);
    } else {
        name = in;
        tag = "";
    }
}

ModelRef parse_model_ref(const std::string & raw) {
    std::string input = trim(raw);
    if (input.empty()) {
        fail("empty model name");
    }

    ModelRef ref;

    if (looks_like_path(input)) {
        ref.source = ModelRef::FILE;
        ref.file = expand_tilde(input);
        return ref;
    }

    bool forced_hf = false;
    bool forced_ollama = false;
    for (const char * p : { "hf:", "hf.co/", "huggingface.co/", "https://huggingface.co/" }) {
        if (starts_with(input, p)) {
            input = input.substr(strlen(p));
            forced_hf = true;
            break;
        }
    }
    if (!forced_hf && starts_with(input, "ollama:")) {
        input = input.substr(7);
        forced_ollama = true;
    }

    std::string name_part, tag;
    split_tag(input, name_part, tag);
    auto parts = split(name_part, '/');
    for (const auto & p : parts) {
        if (trim(p).empty()) {
            fail("invalid model reference: '" + raw + "'");
        }
    }

    if (forced_hf || (!forced_ollama && parts.size() >= 2)) {
        if (parts.size() != 2) {
            fail("Hugging Face references must look like org/repo[:quant] (got '" + raw + "')");
        }
        ref.source = ModelRef::HF;
        ref.ns = parts[0];
        ref.name = parts[1];
        ref.tag = tag; // quant or file selector; empty → default preference
        return ref;
    }

    ref.source = ModelRef::OLLAMA;
    if (parts.size() == 1) {
        ref.ns = "library";
        ref.name = parts[0];
    } else if (parts.size() == 2) {
        ref.ns = parts[0];
        ref.name = parts[1];
    } else {
        fail("Ollama references must look like [user/]name[:tag] (got '" + raw + "')");
    }
    ref.tag = tag.empty() ? "latest" : tag;
    return ref;
}

std::string ModelRef::display() const {
    switch (source) {
        case FILE:
            return file.string();
        case HF: {
            std::string s = "hf:" + ns + "/" + name;
            if (!tag.empty()) {
                s += ":" + tag;
            }
            return s;
        }
        case OLLAMA:
        default: {
            std::string s = (ns == "library") ? name : ns + "/" + name;
            if (tag != "latest") {
                s += ":" + tag;
            }
            return s;
        }
    }
}

fs::path ModelRef::store_dir(const fs::path & root) const {
    switch (source) {
        case HF:
            return root / "hf" / sanitize_component(ns) / sanitize_component(name) /
                   sanitize_component(tag.empty() ? "default" : tag);
        case OLLAMA:
            return root / "ollama" / sanitize_component(ns) / sanitize_component(name) /
                   sanitize_component(tag);
        case FILE:
        default:
            fail("file references have no store directory");
    }
}

// ---- store --------------------------------------------------------------

fs::path models_root() {
    return alpacca_home() / "models";
}

static std::string read_file_text(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool load_manifest(const fs::path & dir, JValue & out) {
    std::string text = read_file_text(dir / "manifest.json");
    if (text.empty()) {
        return false;
    }
    std::string err;
    return json_parse(text, out, err);
}

std::optional<LocalModel> find_local(const ModelRef & ref) {
    LocalModel m;

    if (ref.source == ModelRef::FILE) {
        std::error_code ec;
        if (!fs::exists(ref.file, ec)) {
            return std::nullopt;
        }
        m.model_path = ref.file;
        return m;
    }

    fs::path dir = ref.store_dir(models_root());
    if (!load_manifest(dir, m.manifest)) {
        return std::nullopt;
    }
    std::string model_file = m.manifest.get_str("model_file");
    if (model_file.empty()) {
        return std::nullopt;
    }
    m.dir = dir;
    m.model_path = dir / model_file;
    std::error_code ec;
    if (!fs::exists(m.model_path, ec)) {
        return std::nullopt;
    }
    std::string mmproj = m.manifest.get_str("mmproj_file");
    if (!mmproj.empty() && fs::exists(dir / mmproj, ec)) {
        m.mmproj_path = dir / mmproj;
    }
    return m;
}

void write_manifest(const fs::path & dir, const JValue & manifest) {
    fs::create_directories(dir);
    fs::path tmp = dir / "manifest.json.tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            fail("cannot write " + tmp.string());
        }
        f << manifest.dump(2) << "\n";
    }
    fs::rename(tmp, dir / "manifest.json");
}

std::vector<ListedModel> list_models() {
    std::vector<ListedModel> out;
    fs::path root = models_root();
    std::error_code ec;
    if (!fs::exists(root, ec)) {
        return out;
    }

    for (auto it = fs::recursive_directory_iterator(root, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            break;
        }
        if (!it->is_regular_file(ec) || it->path().filename() != "manifest.json") {
            continue;
        }
        fs::path dir = it->path().parent_path();
        JValue man;
        if (!load_manifest(dir, man)) {
            continue;
        }
        ListedModel lm;
        lm.dir = dir;
        lm.name = man.get_str("name", dir.filename().string());
        lm.source = man.get_str("source", "?");
        lm.size = (uint64_t) man.get_num("size", 0);
        if (lm.size == 0) {
            for (auto & f : fs::directory_iterator(dir, ec)) {
                if (f.is_regular_file(ec) && f.path().extension() == ".gguf") {
                    lm.size += (uint64_t) f.file_size(ec);
                }
            }
        }
        lm.modified = man.get_str("pulled_at", "");
        out.push_back(std::move(lm));
    }

    std::sort(out.begin(), out.end(), [](const ListedModel & a, const ListedModel & b) {
        return a.name < b.name;
    });
    return out;
}

bool remove_model(const ModelRef & ref) {
    if (ref.source == ModelRef::FILE) {
        fail("refusing to delete a raw file path; remove it yourself if intended");
    }
    fs::path dir = ref.store_dir(models_root());
    std::error_code ec;
    if (!fs::exists(dir / "manifest.json", ec)) {
        return false;
    }
    fs::remove_all(dir, ec);
    if (ec) {
        fail("failed to remove " + dir.string() + ": " + ec.message());
    }
    // Tidy now-empty parents up to the models root.
    fs::path parent = dir.parent_path();
    while (parent != models_root() && fs::is_empty(parent, ec) && !ec) {
        fs::remove(parent, ec);
        parent = parent.parent_path();
    }
    return true;
}

} // namespace alpacca
