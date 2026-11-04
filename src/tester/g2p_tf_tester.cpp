/* 
 *
 * Copyright 2026 Alan Crispin <crispinalan@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * g2p_tf_tester.cpp
 * Pure C++17 Transformer-based G2P Inference Engine
 *
 * This file implements a transformer model for grapheme-to-phoneme conversion.
 * It loads a pre-trained model and performs inference on input words to generate phonemes.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <numeric>
#include <cassert>

/** @brief Configuration constants for the transformer model */
constexpr int D_MODEL = 64;        ///< Dimension of model embeddings
constexpr int NUM_HEADS = 4;       ///< Number of attention heads
constexpr int D_FF = 128;          ///< Dimension of feed-forward network
constexpr int NUM_LAYERS = 2;      ///< Number of transformer layers
constexpr int MAX_SRC_LEN = 30;    ///< Maximum source sequence length
constexpr int MAX_TGT_LEN = 40;    ///< Maximum target sequence length

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

    /** @brief Access element at row r, column c
     *  @param r Row index
     *  @param c Column index
     *  @return Reference to the element
     */
    double& at(int r, int c) { return data[r * cols + c]; }

    /** @brief Const access element at row r, column c
     *  @param r Row index
     *  @param c Column index
     *  @return Const reference to the element
     */
    const double& at(int r, int c) const { return data[r * cols + c]; }
};

/** @brief Transpose a matrix
 *  @param A Input matrix
 *  @return Transposed matrix
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

/** @brief Apply softmax to each row of a matrix
 *  @param m Input matrix
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

/** @brief Transformer model for grapheme-to-phoneme conversion */
struct Transformer {
    Mat src_emb, tgt_emb, pos_enc_src, pos_enc_tgt;
    std::vector<Mat> enc_q, enc_k, enc_v, enc_out, enc_ff1, enc_ff2;
    std::vector<Mat> dec_q, dec_k, dec_v, dec_out, dec_cross_q, dec_cross_k, dec_cross_v, dec_cross_out, dec_ff1, dec_ff2;
    Mat out_proj;

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

        // Compute Position Encodings explicitly
        for (int i = 0; i < MAX_SRC_LEN; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                pos_enc_src.at(i, j) = sin(i / pow(10000.0, 2.0 * (j / 2) / D_MODEL));
        for (int i = 0; i < MAX_TGT_LEN; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                pos_enc_tgt.at(i, j) = sin(i / pow(10000.0, 2.0 * (j / 2) / D_MODEL));
    }

    /** @brief Load model weights from file
     *  @param path File path to load weights from
     *  @return true if successful, false otherwise
     */
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        auto read = [&](Mat& m) { f.read((char*)m.data.data(), m.data.size() * sizeof(double)); };

        // Match the layer-by-layer flat mapping used in the trainer
        for (int l = 0; l < NUM_LAYERS; ++l) {
            read(enc_q[l]);
            read(enc_k[l]);
            read(enc_v[l]);
            read(enc_out[l]);
            read(enc_ff1[l]);
            read(enc_ff2[l]);
            read(dec_q[l]);
            read(dec_k[l]);
            read(dec_v[l]);
            read(dec_out[l]);
            read(dec_cross_q[l]);
            read(dec_cross_k[l]);
            read(dec_cross_v[l]);
            read(dec_cross_out[l]);
            read(dec_ff1[l]);
            read(dec_ff2[l]);
        }
        read(out_proj);
        return true;
    }

    /** @brief Forward pass through the transformer
     *  @param src Source sequence IDs
     *  @param tgt Target sequence IDs (for auto-regressive decoding)
     *  @return Output logits
     */
    Mat forward(const std::vector<int>& src, const std::vector<int>& tgt) const {
        int sl = src.size(), tl = tgt.size();
        if (sl > MAX_SRC_LEN) sl = MAX_SRC_LEN;
        if (tl > MAX_TGT_LEN) tl = MAX_TGT_LEN;

        Mat x(sl, D_MODEL), y(tl, D_MODEL);
        for (int i = 0; i < sl; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                x.at(i, j) = src_emb.at(src[i], j) + pos_enc_src.at(i, j);
        for (int i = 0; i < tl; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                y.at(i, j) = tgt_emb.at(tgt[i], j) + pos_enc_tgt.at(i, j);

        // Encoder Blocks
        for (int l = 0; l < NUM_LAYERS; ++l) {
            Mat q = mat_mul(x, enc_q[l]), k = mat_mul(x, enc_k[l]), v = mat_mul(x, enc_v[l]);

            // Self-attention logic
            Mat kT = mat_transpose(k);
            Mat scores = mat_mul(q, kT);
            for (auto& val : scores.data) val /= std::sqrt(D_MODEL);
            softmax_row(scores);

            Mat att_out = mat_mul(scores, v);
            Mat out = mat_add(mat_mul(att_out, enc_out[l]), x);

            // Layer Norm 1
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            // Feed Forward Network (FFN)
            Mat ffn = mat_mul(out, enc_ff1[l]);
            for (int j = 0; j < ffn.cols; ++j)
                ffn.at(0, j) = std::max(0.0, ffn.at(0, j)); // ReLU
            ffn = mat_mul(ffn, enc_ff2[l]);
            out = mat_add(out, ffn);

            // Layer Norm 2
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    m += out.at(i, j);
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

        // Decoder Blocks
        for (int l = 0; l < NUM_LAYERS; ++l) {
            // Masked Decoder Self-Attention
            Mat q = mat_mul(y, dec_q[l]), k = mat_mul(y, dec_k[l]), v = mat_mul(y, dec_v[l]);
            Mat kT = mat_transpose(k);
            Mat scores = mat_mul(q, kT);
            for (auto& val : scores.data) val /= std::sqrt(D_MODEL);
            for (int i = 0; i < scores.rows; ++i)
                for (int j = i + 1; j < scores.cols; ++j)
                    scores.at(i, j) = -1e9; // Causal masking
            softmax_row(scores);

            Mat att_out = mat_mul(scores, v);
            Mat out = mat_add(mat_mul(att_out, dec_out[l]), y);

            // Layer Norm 1
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            // Cross Attention
            q = mat_mul(out, dec_cross_q[l]);
            k = mat_mul(enc_out, dec_cross_k[l]);
            v = mat_mul(enc_out, dec_cross_v[l]);
            kT = mat_transpose(k);
            scores = mat_mul(q, kT);
            for (auto& val : scores.data) val /= std::sqrt(D_MODEL);
            softmax_row(scores);
            att_out = mat_mul(scores, v);
            out = mat_add(mat_mul(att_out, dec_cross_out[l]), out);

            // Layer Norm 2
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    m += out.at(i, j);
                m /= out.cols;
                double s = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    s += (out.at(i, j) - m) * (out.at(i, j) - m);
                s = std::sqrt(s / out.cols + 1e-9);
                for (int j = 0; j < out.cols; ++j)
                    out.at(i, j) = (out.at(i, j) - m) / s;
            }

            // FFN
            Mat ffn = mat_mul(out, dec_ff1[l]);
            for (int j = 0; j < ffn.cols; ++j)
                ffn.at(0, j) = std::max(0.0, ffn.at(0, j));
            ffn = mat_mul(ffn, dec_ff2[l]);
            out = mat_add(out, ffn);

            // Layer Norm 3
            for (int i = 0; i < out.rows; ++i) {
                double m = 0.0;
                for (int j = 0; j < out.cols; ++j)
                    m += out.at(i, j);
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
        return mat_mul(y, out_proj);
    }
};

/** @brief Greedy decoding to generate phoneme sequence
 *  @param model Transformer model
 *  @param src_ids Source word IDs
 *  @param pt Phoneme tokenizer
 *  @return Generated phoneme sequence
 */
std::vector<std::string> greedy_decode(const Transformer& model, const std::vector<int>& src_ids, const PhonemeTokenizer& pt) {
    std::vector<int> tgt_ids = {pt.encode("<SOS>")};
    while (tgt_ids.back() != pt.encode("<EOS>") && (int)tgt_ids.size() <= MAX_TGT_LEN) {
        Mat logits = model.forward(src_ids, tgt_ids);
        int best = 0;
        double max_v = -1e9;
        int target_row = logits.rows - 1; // Predict the token following the final index
        for (int j = 0; j < logits.cols; ++j) {
            if (logits.at(target_row, j) > max_v) {
                max_v = logits.at(target_row, j);
                best = j;
            }
        }
        tgt_ids.push_back(best);
    }

    std::vector<std::string> res;
    for (int id : tgt_ids) {
        std::string p = pt.decode(id);
        if (p != "<PAD>" && p != "<SOS>" && p != "<EOS>")
            res.push_back(p);
    }
    return res;
}

/** @brief Calculate edit distance between two sequences
 *  @param a First sequence
 *  @param b Second sequence
 *  @return Edit distance (number of operations needed to transform a into b)
 */
int edit_distance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    int n = a.size(), m = b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));
    for (int i = 0; i <= n; ++i)
        dp[i][0] = i;
    for (int j = 0; j <= m; ++j)
        dp[0][j] = j;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            dp[i][j] = std::min({dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (a[i-1] != b[j-1])});
        }
    }
    return dp[n][m];
}

/** @brief Run a test case comparing predicted and expected phonemes
 *  @param model Transformer model
 *  @param ct Character tokenizer
 *  @param pt Phoneme tokenizer
 *  @param word Input word to test
 *  @param expected Expected phoneme sequence
 */
void run_test(const Transformer& model, const CharTokenizer& ct, const PhonemeTokenizer& pt, const std::string& word, const std::vector<std::string>& expected) {
    std::vector<int> src;
    src.push_back(ct.encode("<SOS>"));
    for (char c : word)
        src.push_back(ct.encode(std::tolower(static_cast<unsigned char>(c))));
    src.push_back(ct.encode("<EOS>"));

    auto predicted = greedy_decode(model, src, pt);

    bool pass = (predicted.size() == expected.size());
    if (pass) {
        for (size_t i = 0; i < predicted.size(); ++i)
            if (predicted[i] != expected[i]) {
                pass = false;
                break;
            }
    }

    int wer = edit_distance(predicted, expected);
    double cer = (double)wer / std::max(1, (int)expected.size());

    std::cout << std::setw(10) << word << ": "
              << (pass ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
              << " | WER: " << wer << " | CER: " << std::fixed << std::setprecision(2) << cer
              << " | Pred: ";
    for (size_t i = 0; i < predicted.size(); ++i)
        std::cout << predicted[i] << (i+1 < predicted.size() ? " " : "");
    std::cout << "\n";
}

/** @brief Main function - loads model and runs test cases */
int main() {
    CharTokenizer ct;
    PhonemeTokenizer pt;
    Transformer model(ct.vocab_size(), pt.vocab_size());
    if (!model.load("g2p_tf_model")) {
        std::cerr << "Error: Cannot load g2p_tf_model. Train first.\n";
        return 1;
    }
    std::cout << "Model loaded. Running inference...\n";

    // Test cases
    run_test(model, ct, pt, "cat", {"K", "AE", "T"});
    run_test(model, ct, pt, "knife", {"N", "AY", "F"});
    run_test(model, ct, pt, "cake", {"K", "EY", "K"});
    run_test(model, ct, pt, "phone", {"F", "OW", "N"});
    run_test(model, ct, pt, "through", {"TH", "R", "UW"});
    run_test(model, ct, pt, "strength", {"S", "T", "R", "EH", "NG", "TH"});
    run_test(model, ct, pt, "queue", {"K", "Y", "UW"});
    run_test(model, ct, pt, "rhythm", {"R", "IH", "DH", "M"});

    return 0;
}
