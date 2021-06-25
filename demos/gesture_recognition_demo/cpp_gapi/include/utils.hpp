#include <opencv2/core.hpp>

// Take from user`s input
const float AR_THRASHOLD = 0.8;

const int   WAITING_PERSON_DURATION = 8;
const float IOU_THRESHOLD = 0.3;
const int NUM_CLASSES = 100;
const float BOUNDING_BOX_THRESHOLD = 0.4;
// Extract that thing from person detection net
const float PERSON_DETECTOR_W = 320.;
const float PERSON_DETECTOR_H = 320.;

struct rectHash{
    std::size_t operator()(const cv::Rect& rect) const {
        std::hash<int> hashVal;
        return hashVal(static_cast<int>(floor(rect.x / rect.width) + floor(rect.y / rect.height)));
  }
};

struct rectEqual{
    bool operator() (const cv::Rect& l, const cv::Rect& r) const {
        return l.x == r.x && l.y == r.y &&
               l.width == r.width &&
               l.height == r.height;
    }
};

struct Detection
{
    cv::Rect roi;
    int waiting = 1;

    Detection(const cv::Rect& r = cv::Rect(), const int w = 0):
        roi(r), waiting(w){}
};

struct RegisteredPersons {
    std::map<size_t, Detection> active_persons;
    std::map<size_t, Detection> waiting_persons;
    int last_id = 0;
};

using BoundingBoxesSet = std::unordered_set<cv::Rect, rectHash, rectEqual>;

void setInput(cv::GStreamingCompiled stream, const std::string& input ) {
    try {
        // If stoi() throws exception input should be a path not a camera id
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(std::stoi(input)));
    } catch (std::invalid_argument&) {
        slog::info << "Input source is treated as a file path" << slog::endl;
        stream.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));
    }
}

static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

// Take it from user`s json
const std::vector<std::string> ACTIONS_MAP  =  {"hello",
                                                "nice",
                                                "teacher",
                                                "eat",
                                                "no",
                                                "happy",
                                                "like",
                                                "orange",
                                                "want",
                                                "deaf",
                                                "school",
                                                "sister",
                                                "finish",
                                                "white",
                                                "bird",
                                                "what",
                                                "tired",
                                                "friend",
                                                "sit",
                                                "mother",
                                                "yes",
                                                "student",
                                                "learn",
                                                "spring",
                                                "good",
                                                "fish",
                                                "again",
                                                "sad",
                                                "table",
                                                "need",
                                                "where",
                                                "father",
                                                "milk",
                                                "cousin",
                                                "brother",
                                                "paper",
                                                "forget",
                                                "nothing",
                                                "book",
                                                "girl",
                                                "fine",
                                                "black",
                                                "boy",
                                                "lost",
                                                "family",
                                                "hearing",
                                                "bored",
                                                "please",
                                                "water",
                                                "computer",
                                                "help",
                                                "doctor",
                                                "yellow",
                                                "write",
                                                "hungry",
                                                "but",
                                                "drink",
                                                "bathroom",
                                                "man",
                                                "how",
                                                "understand",
                                                "red",
                                                "beautiful",
                                                "sick",
                                                "blue",
                                                "green",
                                                "english",
                                                "name",
                                                "you",
                                                "who",
                                                "same",
                                                "nurse",
                                                "day",
                                                "now",
                                                "brown",
                                                "thanks",
                                                "hurt",
                                                "here",
                                                "grandmother",
                                                "pencil",
                                                "walk",
                                                "bad",
                                                "read",
                                                "when",
                                                "dance",
                                                "play",
                                                "sign",
                                                "go",
                                                "big",
                                                "sorry",
                                                "work",
                                                "draw",
                                                "grandfather",
                                                "woman",
                                                "right",
                                                "france",
                                                "pink",
                                                "know",
                                                "live",
                                                "night"};

