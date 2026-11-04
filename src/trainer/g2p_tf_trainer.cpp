/*
 * @file g2p_tf_trainer.cpp
 * @brief Pure C++17 Transformer-based Grapheme-to-Phoneme Trainer
 *
 * This file implements a complete transformer model for grapheme-to-phoneme conversion.
 * It includes all components needed for training: data loading, model definition,
 * optimization, and saving the trained model.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cctype>
#include <numeric>
#include <stdexcept>
#include <cassert>

/** @brief Configuration constants for the transformer model */
constexpr int D_MODEL = 64;        ///< Dimension of model embeddings
constexpr int NUM_HEADS = 4;       ///< Number of attention heads
constexpr int D_FF = 128;          ///< Dimension of feed-forward network
constexpr int NUM_LAYERS = 2;      ///< Number of transformer layers
constexpr int MAX_SRC_LEN = 30;    ///< Maximum source sequence length
constexpr int MAX_TGT_LEN = 40;    ///< Maximum target sequence length
constexpr int BATCH_SIZE = 8;      ///< Batch size for training (currently unused)
constexpr int EPOCHS = 500;        ///< Number of training epochs
constexpr double LEARNING_RATE = 1e-3;  ///< Learning rate
constexpr double WEIGHT_DECAY = 0.01;    ///< Weight decay for regularization
constexpr double DROPOUT = 0.15;     ///< Dropout rate (currently unused)
constexpr double GRAD_CLIP = 1.0;   ///< Gradient clipping threshold
constexpr int EARLY_STOP_PATIENCE = 35;  ///< Early stopping patience
constexpr int MAX_TRAINING_WORDS = 1000;  ///< Maximum number of words to train on

/** @brief Matrix structure for tensor operations */
struct Mat {
    int rows, cols;
    std::vector<double> data;

    /** @brief Default constructor */
    Mat() : rows(0), cols(0) {}

    /** @brief Constructor with dimensions and optional initialization value
     *  @param r Number of rows
     *  @param c Number of columns
     *  @param init Initialization value (default: 0.0)
     */
    Mat(int r, int c, double init = 0.0) : rows(r), cols(c), data(r * c, init) {}

    /** @brief Access element at row r, column c with bounds checking
     *  @param r Row index
     *  @param c Column index
     *  @return Reference to the element
     */
    double& at(int r, int c) {
        assert(r >= 0 && r < rows && c >= 0 && c < cols);
        return data[r * cols + c];
    }

    /** @brief Const access element at row r, column c with bounds checking
     *  @param r Row index
     *  @param c Column index
     *  @return Const reference to the element
     */
    const double& at(int r, int c) const {
        assert(r >= 0 && r < rows && c >= 0 && c < cols);
        return data[r * cols + c];
    }

    /** @brief Zero out all elements of the matrix */
    void zero() { std::fill(data.begin(), data.end(), 0.0); }
};

/** @brief Transpose a matrix
 *  @param A Input matrix (rows × cols)
 *  @return Transposed matrix (cols × rows)
 */
Mat mat_transpose(const Mat& A) {
    Mat B(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            B.at(j, i) = A.at(i, j);
    return B;
}

/** @brief Matrix multiplication
 *  @param A First matrix (rows × cols)
 *  @param B Second matrix (cols × cols2)
 *  @return Resulting matrix (rows × cols2)
 */
Mat mat_mul(const Mat& A, const Mat& B) {
    assert(A.cols == B.rows);
    Mat C(A.rows, B.cols, 0.0);
    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            double aik = A.at(i, k);
            for (int j = 0; j < B.cols; ++j) {
                C.at(i, j) += aik * B.at(k, j);
            }
        }
    }
    return C;
}

/** @brief Matrix addition
 *  @param A First matrix
 *  @param B Second matrix (same dimensions as A)
 *  @return Resulting matrix (A + B)
 */
Mat mat_add(const Mat& A, const Mat& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    Mat C(A.rows, A.cols);
    for (int i = 0; i < A.rows * A.cols; ++i) C.data[i] = A.data[i] + B.data[i];
    return C;
}

/** @brief Matrix subtraction
 *  @param A First matrix
 *  @param B Second matrix (same dimensions as A)
 *  @return Resulting matrix (A - B)
 */
Mat mat_sub(const Mat& A, const Mat& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    Mat C(A.rows, A.cols);
    for (int i = 0; i < A.rows * A.cols; ++i) C.data[i] = A.data[i] - B.data[i];
    return C;
}

/** @brief Apply softmax to each row of a matrix
 *  @param m Input matrix (rows × cols)
 */
void softmax_row(Mat& m) {
    for (int i = 0; i < m.rows; ++i) {
        double max_v = -1e9;
        for (int j = 0; j < m.cols; ++j) if (m.at(i, j) > max_v) max_v = m.at(i, j);
        double sum = 0.0;
        for (int j = 0; j < m.cols; ++j) {
            m.at(i, j) = exp(m.at(i, j) - max_v);
            sum += m.at(i, j);
        }
        for (int j = 0; j < m.cols; ++j) m.at(i, j) /= (sum + 1e-9);
    }
}

/** @brief Tokenizer for converting characters to IDs */
struct CharTokenizer {
    std::map<std::string, int> enc;
    std::map<int, std::string> dec;

    /** @brief Constructor initializes character mappings */
    CharTokenizer() {
        std::string chars = "abcdefghijklmnopqrstuvwxyz";
        for (int i = 0; i < (int)chars.size(); ++i) {
            std::string s(1, chars[i]);
            enc[s] = i + 1;
            dec[i + 1] = s;
        }
        enc["<PAD>"] = 0;
        dec[0] = "<PAD>";
        enc["<SOS>"] = 27;
        dec[27] = "<SOS>";
        enc["<EOS>"] = 28;
        dec[28] = "<EOS>";
    }

    /** @brief Encode a string to ID
     *  @param s Input string (single character)
     *  @return Corresponding ID
     */
    int encode(const std::string& s) const { return enc.count(s) ? enc.at(s) : 0; }

    /** @brief Encode a character to ID
     *  @param c Input character
     *  @return Corresponding ID
     */
    int encode(char c) const { return encode(std::string(1, c)); }

    /** @brief Decode an ID to string
     *  @param id Input ID
     *  @return Corresponding string
     */
    std::string decode(int id) const { return dec.count(id) ? dec.at(id) : "<PAD>"; }

    /** @brief Get vocabulary size
     *  @return Number of tokens in vocabulary
     */
    int vocab_size() const { return 29; }
};

/** @brief Tokenizer for converting phonemes to IDs */
struct PhonemeTokenizer {
    std::map<std::string, int> enc;
    std::map<int, std::string> dec;

    /** @brief Constructor initializes phoneme mappings */
    PhonemeTokenizer() {
        std::vector<std::string> phs = {
            "AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW",
            "B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P","R","S","SH","T","TH","V","W","Y","Z","ZH","_"
        };
        for (int i = 0; i < (int)phs.size(); ++i) {
            enc[phs[i]] = i + 1;
            dec[i + 1] = phs[i];
        }
        enc["<PAD>"] = 0;
        dec[0] = "<PAD>";
        enc["<SOS>"] = 40;
        dec[40] = "<SOS>";
        enc["<EOS>"] = 41;
        dec[41] = "<EOS>";
    }

    /** @brief Encode a phoneme string to ID
     *  @param p Input phoneme string
     *  @return Corresponding ID
     */
    int encode(const std::string& p) const { return enc.count(p) ? enc.at(p) : 0; }

    /** @brief Decode an ID to phoneme string
     *  @param id Input ID
     *  @return Corresponding phoneme string
     */
    std::string decode(int id) const { return dec.count(id) ? dec.at(id) : "<PAD>"; }

    /** @brief Get vocabulary size
     *  @return Number of tokens in vocabulary
     */
    int vocab_size() const { return 42; }
};

/** @brief Training sample containing source and target sequences */
struct Sample {
    std::vector<int> src_ids;
    std::vector<int> tgt_ids;
};

/** @brief Remove stress marks from phoneme strings
 *  @param p Input phoneme string with stress marks
 *  @return Phoneme string without stress marks
 */
std::string strip_stress(const std::string& p) {
    std::string s = p;
    while (!s.empty() && std::isdigit(s.back())) s.pop_back();
    return s.empty() ? "_" : s;
}

/** @brief Load and align training data from dictionary file
 *  @param dict_path Path to CMU dictionary file
 *  @param ct Character tokenizer
 *  @param pt Phoneme tokenizer
 *  @return Vector of aligned samples (limited to MAX_TRAINING_WORDS)
 */
std::vector<Sample> load_and_align(const std::string& dict_path, const CharTokenizer& ct, const PhonemeTokenizer& pt) {
    std::ifstream f(dict_path);
    if (!f) throw std::runtime_error("Cannot open dictionary");
    std::vector<Sample> samples;
    std::string line;
    int count = 0;
    while (std::getline(f, line) && count < MAX_TRAINING_WORDS) {
        if (line.empty() || line[0] == ';') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string word = line.substr(0, sp);
        std::string ph_str = line.substr(sp + 2);
        std::string lw;
        for (char c : word) lw += std::tolower(static_cast<unsigned char>(c));
        std::vector<std::string> phs;
        std::string tok;
        for (char c : ph_str) {
            if (c == ' ') { if (!tok.empty()) phs.push_back(strip_stress(tok)); tok.clear(); }
            else tok += c;
        }
        if (!tok.empty()) phs.push_back(strip_stress(tok));
        if (phs.empty()) continue;
        std::vector<int> src, tgt;
        src.push_back(ct.encode("<SOS>"));
        for (size_t i = 0; i < lw.size() && src.size() < MAX_SRC_LEN - 1; ++i)
            src.push_back(ct.encode(lw[i]));
        src.push_back(ct.encode("<EOS>"));
        tgt.push_back(pt.encode("<SOS>"));
        for (size_t i = 0; i < lw.size() && tgt.size() < MAX_TGT_LEN - 1; ++i) {
            int pidx = (int)((i * phs.size()) / lw.size());
            if (i == lw.size() - 1) pidx = phs.size() - 1;
            tgt.push_back(pt.encode(phs[pidx]));
        }
        tgt.push_back(pt.encode("<EOS>"));
        samples.push_back({src, tgt});
        ++count;
    }
    std::cout << "Loaded " << samples.size() << " samples (max: " << MAX_TRAINING_WORDS << ").\n";
    return samples;
}

/** @brief Transformer model for grapheme-to-phoneme conversion */
struct Transformer {
    Mat src_emb, tgt_emb;
    Mat pos_enc_src, pos_enc_tgt;
    std::vector<Mat> enc_q, enc_k, enc_v, enc_out;
    std::vector<Mat> enc_ff1, enc_ff2;
    std::vector<Mat> dec_q, dec_k, dec_v, dec_out;
    std::vector<Mat> dec_cross_q, dec_cross_k, dec_cross_v, dec_cross_out;
    std::vector<Mat> dec_ff1, dec_ff2;
    Mat out_proj;

    /** @brief Cache structure for storing intermediate values during forward pass */
    struct Cache {
        Mat x, y, logits;
        std::vector<Mat> enc_out, dec_out;
        std::vector<Mat> enc_attn, dec_attn, cross_attn;
        std::vector<Mat> enc_ffn, dec_ffn;
        std::vector<Mat> enc_ln_in, dec_ln_in, cross_ln_in, ffn_ln_in;
    };

    /** @brief Constructor initializes model parameters
     *  @param src_vocab Source vocabulary size
     *  @param tgt_vocab Target vocabulary size
     */
    Transformer(int src_vocab, int tgt_vocab) {
        src_emb = Mat(src_vocab, D_MODEL);
        tgt_emb = Mat(tgt_vocab, D_MODEL);
        pos_enc_src = Mat(MAX_SRC_LEN, D_MODEL);
        pos_enc_tgt = Mat(MAX_TGT_LEN, D_MODEL);

        for (int l = 0; l < NUM_LAYERS; ++l) {
            enc_q.push_back(Mat(D_MODEL, D_MODEL));
            enc_k.push_back(Mat(D_MODEL, D_MODEL));
            enc_v.push_back(Mat(D_MODEL, D_MODEL));
            enc_out.push_back(Mat(D_MODEL, D_MODEL));
            enc_ff1.push_back(Mat(D_MODEL, D_FF));
            enc_ff2.push_back(Mat(D_FF, D_MODEL));

            dec_q.push_back(Mat(D_MODEL, D_MODEL));
            dec_k.push_back(Mat(D_MODEL, D_MODEL));
            dec_v.push_back(Mat(D_MODEL, D_MODEL));
            dec_out.push_back(Mat(D_MODEL, D_MODEL));
            dec_cross_q.push_back(Mat(D_MODEL, D_MODEL));
            dec_cross_k.push_back(Mat(D_MODEL, D_MODEL));
            dec_cross_v.push_back(Mat(D_MODEL, D_MODEL));
            dec_cross_out.push_back(Mat(D_MODEL, D_MODEL));
            dec_ff1.push_back(Mat(D_MODEL, D_FF));
            dec_ff2.push_back(Mat(D_FF, D_MODEL));
        }
        out_proj = Mat(D_MODEL, tgt_vocab);
        init_params();
    }

    /** @brief Initialize model parameters with random values */
    void init_params() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> d(0, 0.02);

        auto init = [&](Mat& m) { for (auto& v : m.data) v = d(gen); };

        // Initialize all weight matrices
        for (int l = 0; l < NUM_LAYERS; ++l) {
            init(enc_q[l]);
            init(enc_k[l]);
            init(enc_v[l]);
            init(enc_out[l]);
            init(enc_ff1[l]);
            init(enc_ff2[l]);

            init(dec_q[l]);
            init(dec_k[l]);
            init(dec_v[l]);
            init(dec_out[l]);
            init(dec_cross_q[l]);
            init(dec_cross_k[l]);
            init(dec_cross_v[l]);
            init(dec_cross_out[l]);
            init(dec_ff1[l]);
            init(dec_ff2[l]);
        }
        init(out_proj);

        // Initialize positional encodings
        for (int i = 0; i < MAX_SRC_LEN; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                pos_enc_src.at(i, j) = sin(i / pow(10000.0, 2.0 * (j / 2) / D_MODEL));
        for (int i = 0; i < MAX_TGT_LEN; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                pos_enc_tgt.at(i, j) = sin(i / pow(10000.0, 2.0 * (j / 2) / D_MODEL));
    }

    /** @brief Forward pass through the transformer
     *  @param src Source sequence IDs
     *  @param tgt Target sequence IDs
     *  @param cache Intermediate values cache
     *  @return Output logits
     */
    Mat forward(const std::vector<int>& src, const std::vector<int>& tgt, Cache& cache) {
        int sl = std::min((int)src.size(), MAX_SRC_LEN);
        int tl = std::min((int)tgt.size(), MAX_TGT_LEN);

        // Initialize input embeddings with positional encodings
        cache.x = Mat(sl, D_MODEL);
        cache.y = Mat(tl, D_MODEL);
        for (int i = 0; i < sl; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                cache.x.at(i, j) = src_emb.at(src[i], j) + pos_enc_src.at(i, j);
        for (int i = 0; i < tl; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                cache.y.at(i, j) = tgt_emb.at(tgt[i], j) + pos_enc_tgt.at(i, j);

        Mat x = cache.x;

        // Encoder blocks
        for (int l = 0; l < NUM_LAYERS; ++l) {
            cache.enc_ln_in.push_back(x);

            // Self-attention
            Mat q = mat_mul(x, enc_q[l]);
            Mat k = mat_mul(x, enc_k[l]);
            Mat v = mat_mul(x, enc_v[l]);

            Mat kT = mat_transpose(k);
            Mat scores = mat_mul(q, kT);
            for (auto& val : scores.data) val /= std::sqrt(D_MODEL);
            softmax_row(scores);

            cache.enc_attn.push_back(scores);
            Mat att_out = mat_mul(scores, v);
            Mat out = mat_add(mat_mul(att_out, enc_out[l]), x);

            // Layer normalization 1
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j) m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            cache.enc_out.push_back(out);

            // Feed-forward network
            Mat ffn = mat_mul(out, enc_ff1[l]);
            for (int j = 0; j < ffn.cols; ++j)
                ffn.at(0, j) = std::max(0.0, ffn.at(0, j)); // ReLU
            cache.enc_ffn.push_back(ffn);

            ffn = mat_mul(ffn, enc_ff2[l]);
            out = mat_add(out, ffn);

            // Layer normalization 2
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j) m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            x = out;
        }

        Mat enc_out = x;
        Mat y = cache.y;

        // Decoder blocks
        for (int l = 0; l < NUM_LAYERS; ++l) {
            cache.dec_ln_in.push_back(y);

            // Masked self-attention
            Mat q = mat_mul(y, dec_q[l]);
            Mat k = mat_mul(y, dec_k[l]);
            Mat v = mat_mul(y, dec_v[l]);

            Mat kT = mat_transpose(k);
            Mat scores = mat_mul(q, kT);
            for (auto& val : scores.data) val /= std::sqrt(D_MODEL);

            // Apply causal mask
            for (int i = 0; i < scores.rows; ++i)
                for (int j = i + 1; j < scores.cols; ++j)
                    scores.at(i, j) = -1e9;

            softmax_row(scores);
            cache.dec_attn.push_back(scores);

            Mat att_out = mat_mul(scores, v);
            Mat out = mat_add(mat_mul(att_out, dec_out[l]), y);

            // Layer normalization 1
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j) m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            cache.cross_ln_in.push_back(out);

            // Cross attention
            q = mat_mul(out, dec_cross_q[l]);
            k = mat_mul(enc_out, dec_cross_k[l]);
            v = mat_mul(enc_out, dec_cross_v[l]);

            kT = mat_transpose(k);
            scores = mat_mul(q, kT);
            for (auto& val : scores.data) val /= std::sqrt(D_MODEL);
            softmax_row(scores);

            cache.cross_attn.push_back(scores);
            att_out = mat_mul(scores, v);
            out = mat_add(mat_mul(att_out, dec_cross_out[l]), out);

            // Layer normalization 2
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j) m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            cache.dec_out.push_back(out);

            // Feed-forward network
            Mat ffn = mat_mul(out, dec_ff1[l]);
            for (int j = 0; j < ffn.cols; ++j)
                ffn.at(0, j) = std::max(0.0, ffn.at(0, j)); // ReLU
            cache.dec_ffn.push_back(ffn);

            ffn = mat_mul(ffn, dec_ff2[l]);
            out = mat_add(out, ffn);

            // Layer normalization 3
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j) m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            y = out;
        }

        cache.logits = mat_mul(y, out_proj);
        return cache.logits;
    }

    /** @brief Backward pass to compute gradients
     *  @param tgt Target sequence IDs
     *  @param cache Intermediate values from forward pass
     *  @param grads Gradient matrices for each parameter
     */
    void backward(const std::vector<int>& tgt, Cache& cache, const std::vector<Mat>& grads) {
        int tl = cache.logits.rows;
        int sl = cache.x.rows;
        (void)sl; // Silences unused variable warning cleanly

        double scale = 1.0 / std::sqrt(D_MODEL);

        // 1. CrossEntropy Gradient w.r.t Logits
        Mat d_logits(tl, out_proj.cols, 0.0);
        for (int t = 1; t < tl; ++t) {
            if (tgt[t] == 0) continue;
            double max_l = -1e9;
            for (int k = 0; k < d_logits.cols; ++k)
                if (cache.logits.at(t, k) > max_l) max_l = cache.logits.at(t, k);
            double sum = 0;
            for (int k = 0; k < d_logits.cols; ++k)
                sum += exp(cache.logits.at(t, k) - max_l);
            for (int k = 0; k < d_logits.cols; ++k) {
                double prob = exp(cache.logits.at(t, k) - max_l) / (sum + 1e-9);
                d_logits.at(t, k) = prob - (k == tgt[t] ? 1.0 : 0.0);
            }
        }

        // 2. Output Proj Layer Gradient Step
        int out_proj_index = NUM_LAYERS * 16;
        std::vector<Mat> new_grads = grads; // Create a copy to modify

        new_grads[out_proj_index] = mat_add(new_grads[out_proj_index],
            mat_mul(mat_transpose(cache.dec_out.back()), d_logits));
        Mat d_y = mat_mul(d_logits, mat_transpose(out_proj));

        // 3. Decoder Layer Reversals
        for (int l = NUM_LAYERS - 1; l >= 0; --l) {
            int base = l * 16;
            Mat d_ln = d_y;
            Mat input_dec = (l == 0) ? cache.y : cache.dec_out[l-1];

            // FFN Reverse Pass
            Mat d_ffn = d_ln;
            new_grads[base + 5] = mat_add(new_grads[base + 5],
                mat_mul(mat_transpose(cache.enc_ffn[l]), d_ffn));

            Mat d_ffn_pre = mat_mul(d_ffn, mat_transpose(enc_ff2[l]));
            for (int i = 0; i < d_ffn_pre.rows; ++i)
                for (int j = 0; j < d_ffn_pre.cols; ++j)
                    d_ffn_pre.at(i, j) *= (cache.enc_ffn[l].at(i, j) > 0 ? 1.0 : 0.0);

            new_grads[base + 4] = mat_add(new_grads[base + 4],
                mat_mul(mat_transpose(input_dec), d_ffn_pre));

            // Cross-Attention Analytical Accumulation
            Mat d_cross = d_ln;
            Mat V_cross = mat_mul(cache.enc_out.back(), dec_cross_v[l]);
            Mat K_cross = mat_mul(cache.enc_out.back(), dec_cross_k[l]);
            Mat Q_cross = mat_mul(input_dec, dec_cross_q[l]);

            Mat d_V_cross = mat_mul(mat_transpose(cache.cross_attn[l]), d_cross);
            new_grads[base + 12] = mat_add(new_grads[base + 12],
                mat_mul(mat_transpose(cache.enc_out.back()), d_V_cross));

            Mat d_scores_cross = mat_mul(mat_mul(d_cross, mat_transpose(dec_cross_out[l])), mat_transpose(V_cross));
            for (int i = 0; i < d_scores_cross.rows; ++i)
                for (int j = 0; j < d_scores_cross.cols; ++j)
                    d_scores_cross.at(i, j) *= cache.cross_attn[l].at(i, j) * scale;

            Mat d_Q_cross = mat_mul(d_scores_cross, K_cross);
            new_grads[base + 10] = mat_add(new_grads[base + 10],
                mat_mul(mat_transpose(input_dec), d_Q_cross));

            Mat d_K_cross = mat_mul(mat_transpose(d_scores_cross), Q_cross);
            new_grads[base + 11] = mat_add(new_grads[base + 11],
                mat_mul(mat_transpose(cache.enc_out.back()), d_K_cross));

            d_y = mat_add(mat_mul(d_Q_cross, mat_transpose(dec_cross_q[l])),
                mat_mul(d_K_cross, mat_transpose(dec_cross_k[l])));

            // Self-Attention Back Pass
            Mat d_self = d_y;
            Mat V_self = mat_mul(input_dec, dec_v[l]);
            Mat K_self = mat_mul(input_dec, dec_k[l]);
            Mat Q_self = mat_mul(input_dec, dec_q[l]);

            Mat d_V_self = mat_mul(mat_transpose(cache.dec_attn[l]), d_self);
            new_grads[base + 8] = mat_add(new_grads[base + 8],
                mat_mul(mat_transpose(input_dec), d_V_self));

            Mat d_scores_self = mat_mul(mat_mul(d_self, mat_transpose(dec_out[l])), mat_transpose(V_self));
            for (int i = 0; i < d_scores_self.rows; ++i)
                for (int j = 0; j < d_scores_self.cols; ++j)
                    d_scores_self.at(i, j) *= cache.dec_attn[l].at(i, j) * scale;

            Mat d_Q_self = mat_mul(d_scores_self, K_self);
            new_grads[base + 6] = mat_add(new_grads[base + 6],
                mat_mul(mat_transpose(input_dec), d_Q_self));

            Mat d_K_self = mat_mul(mat_transpose(d_scores_self), Q_self);
            new_grads[base + 7] = mat_add(new_grads[base + 7],
                mat_mul(mat_transpose(input_dec), d_K_self));

            d_y = mat_add(mat_mul(d_Q_self, mat_transpose(dec_q[l])),
                mat_mul(d_K_self, mat_transpose(dec_k[l])));
        }

        // 4. Encoder Layer Reversals
        for (int l = NUM_LAYERS - 1; l >= 0; --l) {
            int base = l * 16;
            Mat d_ln = d_y;
            Mat input_enc = (l == 0) ? cache.x : cache.enc_out[l-1];

            // FFN Reverse Pass
            Mat d_ffn = d_ln;
            new_grads[base + 5] = mat_add(new_grads[base + 5],
                mat_mul(mat_transpose(cache.enc_ffn[l]), d_ffn));

            Mat d_ffn_pre = mat_mul(d_ffn, mat_transpose(enc_ff2[l]));
            for (int i = 0; i < d_ffn_pre.rows; ++i)
                for (int j = 0; j < d_ffn_pre.cols; ++j)
                    d_ffn_pre.at(i, j) *= (cache.enc_ffn[l].at(i, j) > 0 ? 1.0 : 0.0);

            new_grads[base + 4] = mat_add(new_grads[base + 4],
                mat_mul(mat_transpose(input_enc), d_ffn_pre));

            // Attention Reverse pass
            Mat d_attn = d_ln;
            Mat V_enc = mat_mul(input_enc, enc_v[l]);
            Mat K_enc = mat_mul(input_enc, enc_k[l]);
            Mat Q_enc = mat_mul(input_enc, enc_q[l]);

            Mat d_V_enc = mat_mul(mat_transpose(cache.enc_attn[l]), d_attn);
            new_grads[base + 2] = mat_add(new_grads[base + 2],
                mat_mul(mat_transpose(input_enc), d_V_enc));

            Mat d_scores_enc = mat_mul(mat_mul(d_attn, mat_transpose(enc_out[l])), mat_transpose(V_enc));
            for (int i = 0; i < d_scores_enc.rows; ++i)
                for (int j = 0; j < d_scores_enc.cols; ++j)
                    d_scores_enc.at(i, j) *= cache.enc_attn[l].at(i, j) * scale;

            Mat d_Q_enc = mat_mul(d_scores_enc, K_enc);
            new_grads[base] = mat_add(new_grads[base],
                mat_mul(mat_transpose(input_enc), d_Q_enc));

            Mat d_K_enc = mat_mul(mat_transpose(d_scores_enc), Q_enc);
            new_grads[base + 1] = mat_add(new_grads[base + 1],
                mat_mul(mat_transpose(input_enc), d_K_enc));

            d_y = mat_add(mat_mul(d_Q_enc, mat_transpose(enc_q[l])),
                mat_mul(d_K_enc, mat_transpose(enc_k[l])));
        }
    }
};

/** @brief AdamW optimizer with weight decay */
struct AdamW {
    std::vector<Mat> m;  ///< First moment estimates
    std::vector<Mat> v;  ///< Second moment estimates (uncentered)
    double lr, wd, beta1, beta2, eps;
    int step = 0;

    /** @brief Constructor for AdamW optimizer
     *  @param lr Learning rate
     *  @param wd Weight decay coefficient
     */
    AdamW(double lr, double wd) : lr(lr), wd(wd), beta1(0.9), beta2(0.999), eps(1e-8) {}

    /** @brief Initialize optimizer state
     *  @param params Model parameters to optimize
     */
    void init(const std::vector<Mat>& params) {
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = Mat(params[i].rows, params[i].cols, 0);
            v[i] = Mat(params[i].rows, params[i].cols, 0);
        }
    }

    /** @brief Perform one optimization step
     *  @param params Model parameters to update
     *  @param grads Current gradients
     */
    void step_update(std::vector<Mat>& params, const std::vector<Mat>& grads) {
        ++step;
        double lr_t = lr * std::sqrt(1 - std::pow(beta2, step)) / (1 - std::pow(beta1, step));

        for (size_t i = 0; i < params.size(); ++i) {
            for (int r = 0; r < params[i].rows; ++r)
                for (int c = 0; c < params[i].cols; ++c) {
                    double g = grads[i].at(r, c);

                    // Update biased first moment estimate
                    m[i].at(r, c) = beta1 * m[i].at(r, c) + (1 - beta1) * g;

                    // Update biased second raw moment estimate
                    v[i].at(r, c) = beta2 * v[i].at(r, c) + (1 - beta2) * g * g;

                    // Compute bias-corrected first moment estimate
                    double m_hat = m[i].at(r, c) / (1 - std::pow(beta1, step));

                    // Compute bias-corrected second raw moment estimate
                    double v_hat = v[i].at(r, c) / (1 - std::pow(beta2, step));

                    // Update parameters with weight decay
                    params[i].at(r, c) -= lr_t * (m_hat / (std::sqrt(v_hat) + eps) + wd * params[i].at(r, c));
                }
        }
    }
};

/** @brief Cosine learning rate schedule
 *  @param base_lr Base learning rate
 *  @param epoch Current epoch number
 *  @param max_epochs Total number of epochs
 *  @return Scaled learning rate for current epoch
 */
double cosine_lr(double base_lr, int epoch, int max_epochs) {
    return 0.5 * base_lr * (1 + std::cos(M_PI * epoch / max_epochs));
}

/** @brief Main training function
 *  @param dict_path Path to CMU dictionary file
 */
void train(const std::string& dict_path) {
    CharTokenizer ct;
    PhonemeTokenizer pt;

    // Load and prepare data
    auto samples = load_and_align(dict_path, ct, pt);
    if (samples.empty()) throw std::runtime_error("No valid samples loaded.");

    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(samples.begin(), samples.end(), rng);

    // Split into train and validation sets
    int train_sz = (int)(samples.size() * 0.8);
    int val_sz = std::min((int)samples.size() - train_sz, (int)(samples.size() * 0.2));

    std::vector<Sample> train_data(samples.begin(), samples.begin() + train_sz);
    std::vector<Sample> val_data(samples.begin() + train_sz, samples.begin() + train_sz + val_sz);

    // Initialize model
    Transformer model(ct.vocab_size(), pt.vocab_size());

    // ==================== LAYER-BY-LAYER PARAMETER COLLECTION BLOCK ====================
    std::vector<Mat> params;
    for (int l = 0; l < NUM_LAYERS; ++l) {
        params.push_back(model.enc_q[l]);          // base + 0
        params.push_back(model.enc_k[l]);          // base + 1
        params.push_back(model.enc_v[l]);          // base + 2
        params.push_back(model.enc_out[l]);        // base + 3
        params.push_back(model.enc_ff1[l]);        // base + 4
        params.push_back(model.enc_ff2[l]);        // base + 5
        params.push_back(model.dec_q[l]);          // base + 6
        params.push_back(model.dec_k[l]);          // base + 7
        params.push_back(model.dec_v[l]);          // base + 8
        params.push_back(model.dec_out[l]);        // base + 9
        params.push_back(model.dec_cross_q[l]);    // base + 10
        params.push_back(model.dec_cross_k[l]);    // base + 11
        params.push_back(model.dec_cross_v[l]);    // base + 12
        params.push_back(model.dec_cross_out[l]);  // base + 13
        params.push_back(model.dec_ff1[l]);        // base + 14
        params.push_back(model.dec_ff2[l]);        // base + 15
    }
    params.push_back(model.out_proj);              // Final Index (Index 32)

    // Synchronization lambda mapping modifications to memory variables
    auto sync_params_to_model = [&]() {
        int idx = 0;
        for (int l = 0; l < NUM_LAYERS; ++l) {
            model.enc_q[l]          = params[idx++];
            model.enc_k[l]          = params[idx++];
            model.enc_v[l]          = params[idx++];
            model.enc_out[l]        = params[idx++];
            model.enc_ff1[l]        = params[idx++];
            model.enc_ff2[l]        = params[idx++];
            model.dec_q[l]          = params[idx++];
            model.dec_k[l]          = params[idx++];
            model.dec_v[l]          = params[idx++];
            model.dec_out[l]        = params[idx++];
            model.dec_cross_q[l]    = params[idx++];
            model.dec_cross_k[l]    = params[idx++];
            model.dec_cross_v[l]    = params[idx++];
            model.dec_cross_out[l]  = params[idx++];
            model.dec_ff1[l]        = params[idx++];
            model.dec_ff2[l]        = params[idx++];
        }
        model.out_proj = params[idx++];
    };

    // Initialize optimizer
    AdamW optimizer(LEARNING_RATE, WEIGHT_DECAY);
    optimizer.init(params);

    double best_val_loss = 1e9;
    int patience_counter = 0;

    Transformer::Cache cache;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double lr = cosine_lr(LEARNING_RATE, epoch, EPOCHS);
        std::shuffle(train_data.begin(), train_data.end(), rng);

        double train_loss = 0.0;
        int train_count = 0;

        // Training step
        for (const auto& s : train_data) {
            cache = Transformer::Cache();
            Mat logits = model.forward(s.src_ids, s.tgt_ids, cache);
            double loss = 0.0;

            // Compute cross-entropy loss
            for (int t = 1; t < (int)s.tgt_ids.size(); ++t) {
                if (s.tgt_ids[t] == 0) continue;
                double max_l = -1e9;
                for (int k = 0; k < logits.cols; ++k)
                    if (logits.at(t, k) > max_l) max_l = logits.at(t, k);
                double sum = 0;
                for (int k = 0; k < logits.cols; ++k)
                    sum += exp(logits.at(t, k) - max_l);
                loss -= log(exp(logits.at(t, s.tgt_ids[t]) - max_l) / (sum + 1e-9));
            }

            train_loss += loss;
            train_count++;

            // Compute gradients
            std::vector<Mat> grads;
            for (const auto& p : params)
                grads.emplace_back(p.rows, p.cols, 0.0);
            model.backward(s.tgt_ids, cache, grads);

            // Gradient Clipping Step
            double total_norm = 0;
            for (const auto& g : grads) for (double v : g.data) total_norm += v * v;
            total_norm = std::sqrt(total_norm);
            if (total_norm > GRAD_CLIP) {
                double scale_factor = GRAD_CLIP / (total_norm + 1e-9);
                for (auto& g : grads) for (double& v : g.data) v *= scale_factor;
            }

            // Update parameters
            optimizer.step_update(params, grads);
            sync_params_to_model(); // Sync weights out of params vector to actual model
        }

        double avg_loss = train_count > 0 ? train_loss / train_count : 0.0;
        std::cout << "Epoch " << epoch << "/" << EPOCHS << " | LR: " << lr << " | Train Loss: " << avg_loss << "\n";

        // Validation step
        double val_loss = 0.0;
        int val_count = 0;
        for (const auto& s : val_data) {
            cache = Transformer::Cache();
            Mat logits = model.forward(s.src_ids, s.tgt_ids, cache);
            double loss = 0.0;

            // Compute cross-entropy loss
            for (int t = 1; t < (int)s.tgt_ids.size(); ++t) {
                if (s.tgt_ids[t] == 0) continue;
                double max_l = -1e9;
                for (int k = 0; k < logits.cols; ++k)
                    if (logits.at(t, k) > max_l) max_l = logits.at(t, k);
                double sum = 0;
                for (int k = 0; k < logits.cols; ++k)
                    sum += exp(logits.at(t, k) - max_l);
                loss -= log(exp(logits.at(t, s.tgt_ids[t]) - max_l) / (sum + 1e-9));
            }

            val_loss += loss;
            val_count++;
        }
        double avg_val = val_count > 0 ? val_loss / val_count : 0.0;
        std::cout << "Val Loss: " << avg_val << "\n";

        // Save best model
        if (avg_val < best_val_loss) {
            best_val_loss = avg_val;
            patience_counter = 0;

            std::ofstream out("g2p_tf_model", std::ios::binary);
            for (const auto& p : params)
                out.write((char*)p.data.data(), p.data.size() * sizeof(double));
            out.close();
        } else {
            patience_counter++;
            if (patience_counter >= EARLY_STOP_PATIENCE) {
                std::cout << "Early stopping triggered.\n";
                break;
            }
        }
    }
}

/** @brief Main function
 *  @param argc Number of command-line arguments
 *  @param argv Command-line arguments
 *  @return Exit status (0 for success)
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <cmudict_file>\n";
        return 1;
    }

    try {
        train(argv[1]);
        std::cout << "Training complete. Model saved to g2p_tf_model.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
