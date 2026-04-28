// - PoseAtlas: cache of decoded poses (per token/time)
// - MicroDecoder: tiny MLP-based decoder prototype (quantized-friendly)
// - DecodeScheduler: schedules decode jobs, supports callbacks

// Intended as a reference runtime module to integrate with engine.
// Replace MicroDecoder internals with real NN inference (GPU/SPU) in production.

#include <bits/stdc++.h>
using namespace std;

// Config / Tunables

constexpr int MAX_JOINTS = 48;          // typical skeleton joints (adjustable)
constexpr int LATENT_SIZE = 32;        // size of latent vector (example)
constexpr int POSE_QUAT_COMPONENTS = 4; // quaternion (x,y,z,w)
constexpr int POSE_STRIDE = MAX_JOINTS * POSE_QUAT_COMPONENTS;

constexpr int ATLAS_CAPACITY = 4096;   // number of pose slots in atlas
constexpr int MAX_DECODE_THREADS = 4;  // concurrency (map to SPUs/compute in prod)
constexpr int DECODE_JOB_BATCH = 4;    // batch size for throughput

// Utility types

struct LatentToken {
    // compact representation of latent (quantized or float)
    // In cooker pipeline this would be quantized bytes. Here we use float for prototype.
    array<float, LATENT_SIZE> data{};
    uint32_t tokenID = 0;   // id referring to animation clip
    uint32_t frameIndex = 0; // frame/time index within clip (or sample index)
};

struct Pose {
    // flattened array: joint0 quat(x,y,z,w), joint1 ...
    // store as floats for runtime skinning upload
    vector<float> q; // size = MAX_JOINTS * 4

    Pose() { q.assign(POSE_STRIDE, 0.0f); }
};

struct PoseKey {
    uint32_t tokenID;
    uint32_t frameIndex;
    bool operator==(PoseKey const &o) const {
        return tokenID == o.tokenID && frameIndex == o.frameIndex;
    }
};

// simple hash for PoseKey
struct PoseKeyHash {
    size_t operator()(PoseKey const &k) const noexcept {
        return (size_t(k.tokenID) << 32) ^ size_t(k.frameIndex);
    }
};

/* ----------------------------- Math helpers --------------------------------- */

// minimal quaternion normalization + utility
static inline void quat_normalize(float q[4]) {
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float n = sqrt(x*x+y*y+z*z+w*w);
    if (n > 1e-8f) { x/=n; y/=n; z/=n; w/=n; }
    q[0]=x; q[1]=y; q[2]=z; q[3]=w;
}

// small deterministic rng for scheduling reproducibility
struct DetermRNG {
    uint64_t state;
    DetermRNG(uint64_t s=1469598103934665603ULL) : state(s) {}
    uint32_t nextU32() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return uint32_t(state & 0xffffffffu);
    }
    float nextFloat() { return (nextU32() / float(0xffffffffu)); }
};

/* ----------------------------- MicroDecoder ---------------------------------
   Prototype micro-decoder: tiny MLP that maps a latent -> joint quaternion deltas.
   In production this is replaced by an NN exported from training (quantized).
   Design goals:
   - Small memory footprint
   - Deterministic
   - SIMD / SPU friendly (no dynamic allocations)
----------------------------------------------------------------------------- */

class MicroDecoder {
public:
    MicroDecoder() {
        init_example_weights();
    }

    // decode latent -> Pose (fills outPose)
    void decode(const LatentToken &latent, Pose &outPose) const {
        // Very small MLP: latent (LATENT_SIZE) -> hidden -> produce per-joint 3 params (axis-angle)
        // We'll decode joint local rotation as small axis-angle perturbation from bind (assume identity bind).
        // This is a toy; real decoder uses proper skeleton bases and root motion.
        const int HIDDEN = 64;
        float hidden[HIDDEN];
        // input -> hidden
        for (int h=0; h<HIDDEN; ++h) {
            float s = bias0[h];
            for (int i=0;i<LATENT_SIZE;++i) s += latent.data[i] * w0[h * LATENT_SIZE + i];
            // activation
            hidden[h] = tanhf(s);
        }
        // hidden -> outputs (we'll output 3*MAX_JOINTS floats representing tiny axis-angle)
        // To keep memory bounded we compute joint-wise
        for (int j=0;j<MAX_JOINTS;++j) {
            float ax=0, ay=0, az=0;
            for (int h=0; h<HIDDEN; ++h) {
                int base = (j*3);
                float v = hidden[h] * w1[(base+0)*HIDDEN + h];
                ax += v;
                v = hidden[h] * w1[(base+1)*HIDDEN + h];
                ay += v;
                v = hidden[h] * w1[(base+2)*HIDDEN + h];
                az += v;
            }
            ax += bias1[j*3 + 0];
            ay += bias1[j*3 + 1];
            az += bias1[j*3 + 2];
            // small axis-angle -> quaternion
            float angle = sqrtf(ax*ax + ay*ay + az*az);
            float q[4];
            if (angle < 1e-6f) {
                q[0]=ax; q[1]=ay; q[2]=az; q[3]=1.0f; // approx identity
            } else {
                float invang = 1.0f/angle;
                float nx = ax * invang;
                float ny = ay * invang;
                float nz = az * invang;
                float ca = cosf(angle);
                float sa = sinf(angle);
                q[0] = nx*sa; q[1] = ny*sa; q[2] = nz*sa; q[3] = ca;
            }
            quat_normalize(q);
            // write into pose
            int idx = j*4;
            outPose.q[idx+0] = q[0];
            outPose.q[idx+1] = q[1];
            outPose.q[idx+2] = q[2];
            outPose.q[idx+3] = q[3];
        }
    }

private:
    // tiny example weights (in production these are loaded from file - quantized)
    // For prototype we initialize deterministic pseudo-random small values
    vector<float> w0; // HIDDEN x LATENT_SIZE
    vector<float> bias0; // HIDDEN
    vector<float> w1; // (MAX_JOINTS*3) x HIDDEN
    vector<float> bias1; // MAX_JOINTS*3

    void init_example_weights() {
        const int HIDDEN = 64;
        w0.assign(HIDDEN * LATENT_SIZE, 0.0f);
        bias0.assign(HIDDEN, 0.0f);
        w1.assign(MAX_JOINTS*3 * HIDDEN, 0.0f);
        bias1.assign(MAX_JOINTS*3, 0.0f);

        DetermRNG rng(0xC0FFEEUL);
        for (auto &x : w0) x = (rng.nextFloat()*2.0f - 1.0f) * 0.05f;
        for (auto &b : bias0) b = (rng.nextFloat()*2.0f - 1.0f) * 0.01f;
        for (auto &x : w1) x = (rng.nextFloat()*2.0f - 1.0f) * 0.02f;
        for (auto &b : bias1) b = (rng.nextFloat()*2.0f - 1.0f) * 0.02f;
    }
};

/* ----------------------------- PoseAtlas -----------------------------------
   A simple thread-safe pose cache mapping (tokenID, frameIndex) -> Pose slot
   LRU eviction policy. On miss, engine should schedule a decode job.
----------------------------------------------------------------------------- */

class PoseAtlas {
public:
    PoseAtlas(size_t capacity = ATLAS_CAPACITY) : _capacity(capacity) {}

    // query, returns true if pose present and copies into outPose
    bool query(const PoseKey &key, Pose &outPose) {
        lock_guard<mutex> g(_mutex);
        auto it = _map.find(key);
        if (it == _map.end()) return false;
        // update LRU
        touch(it->first);
        outPose = it->second.pose;
        return true;
    }

    // insert or overwrite
    void insert(const PoseKey &key, Pose &&pose) {
        lock_guard<mutex> g(_mutex);
        if (_map.size() >= _capacity) evict_one();
        Entry e;
        e.pose = std::move(pose);
        e.lastUsed = _clock++;
        _map.emplace(key, std::move(e));
        // ensure quick lookup: map is unordered_map with PoseKeyHash
    }

    // convenience: check presence
    bool contains(const PoseKey &key) {
        lock_guard<mutex> g(_mutex);
        return _map.find(key) != _map.end();
    }

    size_t size() const {
        lock_guard<mutex> g(_mutex);
        return _map.size();
    }

private:
    struct Entry {
        Pose pose;
        uint64_t lastUsed;
    };

    unordered_map<PoseKey, Entry, PoseKeyHash> _map;
    mutable mutex _mutex;
    size_t _capacity;
    uint64_t _clock = 1;

    void touch(const PoseKey &k) {
        auto it = _map.find(k);
        if (it==_map.end()) return;
        it->second.lastUsed = _clock++;
    }

    void evict_one() {
        // simple LRU eviction: linear scan (capacity moderate)
        uint64_t oldest = UINT64_MAX;
        PoseKey oldestKey{0,0};
        for (auto &kv : _map) {
            if (kv.second.lastUsed < oldest) {
                oldest = kv.second.lastUsed;
                oldestKey = kv.first;
            }
        }
        if (oldest != UINT64_MAX) _map.erase(oldestKey);
    }
};

/* ----------------------------- DecodeScheduler ------------------------------
   Schedules decode jobs (simulated async threads). In production this hooks to SPU
   or GPU compute queue. Provides callback when pose is ready.
----------------------------------------------------------------------------- */

class DecodeScheduler {
public:
    using DecodeCallback = function<void(const PoseKey&, bool /*success*/)>;

    DecodeScheduler(const MicroDecoder &decoder, PoseAtlas &atlas)
        : _decoder(decoder), _atlas(atlas), _running(true) {
        for (int i=0;i<MAX_DECODE_THREADS;++i)
            _workers.emplace_back(&DecodeScheduler::workerThread, this);
    }

    ~DecodeScheduler() {
        {
            lock_guard<mutex> g(_queueMutex);
            _running = false;
            _cv.notify_all();
        }
        for (auto &t : _workers) if (t.joinable()) t.join();
    }

    // schedule decode of latent; returns immediately; callback invoked when done
    void scheduleDecode(LatentToken token, DecodeCallback cb) {
        {
            lock_guard<mutex> g(_queueMutex);
            _queue.emplace_back(move(token), move(cb));
        }
        _cv.notify_one();
    }

    // flush remaining jobs (blocking)
    void flush() {
        unique_lock<mutex> g(_queueMutex);
        _cvEmpty.wait(g, [this]() { return _queue.empty(); });
        // we do not stop workers here
    }

private:
    const MicroDecoder &_decoder;
    PoseAtlas &_atlas;

    // job queue
    deque<pair<LatentToken, DecodeCallback>> _queue;
    mutex _queueMutex;
    condition_variable _cv;
    condition_variable _cvEmpty;
    vector<thread> _workers;
    atomic<bool> _running;

    void workerThread() {
        while (_running) {
            pair<LatentToken, DecodeCallback> job;
            {
                unique_lock<mutex> g(_queueMutex);
                _cv.wait(g, [this]() { return !_queue.empty() || !_running; });
                if (!_running && _queue.empty()) break;
                job = move(_queue.front());
                _queue.pop_front();
            }

            // process batch decode: in this toy example we decode single
            PoseKey key{job.first.tokenID, job.first.frameIndex};
            Pose outPose;
            // decoder
            _decoder.decode(job.first, outPose);
            // insert into atlas
            _atlas.insert(key, std::move(outPose));
            // callback notify
            if (job.second) job.second(key, true);

            // notify flush monitor when queue empties
            {
                lock_guard<mutex> g(_queueMutex);
                if (_queue.empty()) _cvEmpty.notify_all();
            }
        }
    }
};

/* ----------------------------- Usage Example ------------------------------- 
   Example main simulates entities requesting animation frames. For a real engine
   integrate:
   - Cooker produces LatentToken streams stored on disk or networked.
   - Scheduler replaced/integrated with SPU jobs or GPU compute.
   - PoseAtlas upload the pose buffer to GPU skinning SSBO / texture.
----------------------------------------------------------------------------- */

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "NMC runtime prototype starting...\n";

    MicroDecoder decoder;
    PoseAtlas atlas(1024);
    DecodeScheduler scheduler(decoder, atlas);
    // Simulate latent tokens produced by cooker (in real: loaded from disk/ASTRA)
    vector<LatentToken> simulatedTokens;
    simulatedTokens.reserve(200);
    DetermRNG rng(0xBEEF1234);
    for (uint32_t t=0; t<50; ++t) {
        for (uint32_t f=0; f<4; ++f) {
            LatentToken lt;
            lt.tokenID = t;
            lt.frameIndex = f;
            for (int i=0;i<LATENT_SIZE;++i) lt.data[i] = rng.nextFloat()*2.0f - 1.0f;
            simulatedTokens.push_back(lt);
        }
    }
    // Simulate game loop: entities require poses; if missing schedule decode
    const int NUM_ENTITIES = 300;
    struct EntityReq { uint32_t id; uint32_t clip; uint32_t frame; };
    vector<EntityReq> entities;
    for (int i=0;i<NUM_ENTITIES;++i) {
        entities.push_back({(uint32_t)i, (uint32_t)(i%50), (uint32_t)(i%4)});
    }
    atomic<int> callbacks=0;
    auto cb = [&](const PoseKey &k, bool ok){
        callbacks.fetch_add(1);
        // in engine: upload atlas pose slot to GPU skinning buffer
    };
    // Request poses: many will miss and trigger decode
    for (auto &e : entities) {
        PoseKey k{e.clip, e.frame};
        Pose p;
        if (!atlas.query(k, p)) {
            // find latent in simulatedTokens
            // (in engine we would directly read latent by token/frame from package)
            for (auto &lt : simulatedTokens) {
                if (lt.tokenID==e.clip && lt.frameIndex==e.frame) {
                    scheduler.scheduleDecode(lt, cb);
                    break;
                }
            }
        } else {
            // use p immediately
        }
    }
    // wait for background decode to finish (in prod: continuous service)
    scheduler.flush();
    // Validate that atlas contains expected entries
    int found=0;
    for (auto &e : entities) {
        PoseKey k{e.clip, e.frame};
        Pose p;
        if (atlas.query(k,p)) ++found;
    }
    cout << "Decoded poses available in atlas: " << found << " / " << NUM_ENTITIES << "\n";
    cout << "Callback count: " << callbacks.load() << "\n";
    // example: get a pose and print joint0 quat
    PoseKey sampleKey{0,0};
    Pose samplePose;
    if (atlas.query(sampleKey, samplePose)) {
        cout << "Sample pose joint0 quat: ";
        cout << samplePose.q[0] << ", " << samplePose.q[1] << ", "
             << samplePose.q[2] << ", " << samplePose.q[3] << "\n";
    }
    cout << "NMC runtime prototype done.\n";
    return 0;
}
