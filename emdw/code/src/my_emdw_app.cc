// patrec headers
#include "prlite_logging.hpp"  // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "clustergraph.hpp"
#include "discretetable.hpp"
#include "emdw.hpp"
#include "lbu_cg.hpp"
#include "messagequeue.hpp"

// standard headers
#include <cassert>  // For assertions like python's `assert(condition)` functionality
#include <cctype>    // toupper
#include <iostream>  // cout, endl, flush, cin, cerr
#include <limits>    // Not Sure
#include <limits>    // Get Minimum double
#include <map>       // Sparse Probabilities for Factor Creation
#include <memory>    // Not Sure
#include <string>    // string
#include <tuple>     // Packaging \Theta for function output
#include <vector>    // Everything

// Not using this but is being used by initialisation code
using namespace std;
using namespace emdw;

// Some Variables That Will Be Used Throughtout Various Functiosn
typedef DiscreteTable<int> DT;
double defProb = 0.0;

// ==================== Helper Functions ====================
// Print Functions
void debugPrint(bool debug, std::string message);
void printFactor(const Factor &factor, std::string_view factorName);
template <typename T>
void print2DArray(std::vector<std::vector<T>> inp,
                  std::string_view arrayName = "2D Array");
template <typename T>
void printVector(std::vector<T> inp,
                 std::string_view vectorName = "Some Vector");
void printDomainDroughtState(bool debug,
                             const rcptr<std::vector<int>> &droughtStateDomain);
void printDomainAttributeRVsDiscerete(
    bool debug, const std::vector<rcptr<std::vector<int>>> &attributeRvDomains);
// Misc
std::vector<std::string> splitByCharacter(const std::string &inputString,
                                          const char &delimiter);
template <typename T>
std::vector<T> findMaxAlongAxis(const std::vector<std::vector<T>> &inp,
                                size_t axis = 0);
template <typename T>
std::vector<T> createRandomVector(const size_t vecSize, const T mean = 0,
                                  const T variance = 1);

// ==================== File Handling ====================
// Input
std::vector<std::vector<int>> readData(std::string_view filePath,
                                       bool hasHeader = true);
// Model Seclection Output
void saveModelSelectionOutput(std::string_view outputPathCSV,
                              const std::vector<std::vector<double>> &results,
                              const size_t mMin);

void processAndSaveModelOutput(
    std::string_view dirPath, ClusterGraph &cg,
    std::map<Idx2, rcptr<Factor>> &msgs,
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<double> &finalPriors,
    const std::vector<std::vector<double>> &finalTrans,
    const std::vector<std::vector<std::vector<double>>> &finalEmissions,
    const std::vector<std::vector<int>> &observedAttributes);

// ==================== Creating Params ====================
std::vector<double> createRandomPriors(const int m, const double mean = 0.0,
                                       const double variance = 1.0);
std::vector<std::vector<double>> createRandomTransitionParams(
    const int m, const double mean = 0.0, const double variance = 1.0);
std::vector<std::vector<std::vector<double>>> createEmissionParams(
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains,
    const size_t m, const double variance = 1.0, const double mean = 0);

// ==================== Model Setup ====================
// ID Creation
std::vector<emdw::RVIdType> createDiscreteRvIds(int numNodes,
                                                uint &runningIdCount);
// Domain Creation
rcptr<std::vector<int>> createDiscreteRvDomain(size_t C);
// Factor Creation
rcptr<Factor> createPriorS1Factor(
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<double> &oldPriors, const emdw::RVIdType &S1id,
    const double &noiseVariance);
std::vector<rcptr<Factor>> createTransitionFactors(
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<std::vector<double>> &transitionMatrix,
    const std::vector<emdw::RVIdType> &droughtStateIds);
std::vector<std::vector<rcptr<Factor>>> createEmissionFactors(
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains,
    const std::vector<std::vector<std::vector<double>>> &emissionProbs);

// ==================== Validation Functions ====================
bool validateRvIds(const bool &debug,
                   const std::vector<emdw::RVIdType> &droughtStateIds,
                   const std::vector<std::vector<emdw::RVIdType>> &attributeIds,
                   const std::vector<std::vector<float>> &observedAttributes);

// ==================== Model Inference ====================
std::pair<ClusterGraph, std::map<Idx2, rcptr<Factor>>> performLBU_LTRIP(
    const rcptr<Factor> &pS1,
    const std::vector<rcptr<Factor>> &transitionFactors,
    const std::vector<std::vector<rcptr<Factor>>> &emissionFactors,
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const std::vector<std::vector<int>> &observedAttributes);
std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<std::vector<std::vector<double>>>>
runModel(const size_t m,
         const std::vector<std::vector<int>> &observedAttributes,
         const size_t maxIters, std::string_view outputDir, const bool debug);

void modelSelection(const std::vector<std::vector<int>> &observedAttributes,
                    const std::pair<size_t, size_t> &mRange,
                    const size_t maxEMIters, const size_t numRestartIters,
                    std::string_view outputPathCSV, const bool debug);

std::pair<std::vector<int>, double> viterbi(
    const std::vector<double> &logPriors,
    const std::vector<std::vector<double>> &logTrans,
    const std::vector<std::vector<double>> &logEmit);

// ==================== Main Function ====================
int main(int, char *argv[]) {
    // NOTE: this activates logging and unit tests
    initLogging(argv[0]);
    prlite::TestCase::runAllTests();

    try {
        //*********************************************************
        // This is some emdw things, just leave as is...
        //*********************************************************
        unsigned seedVal = emdw::randomEngine.getSeedVal();
        cout << seedVal << endl;
        emdw::randomEngine.setSeedVal(seedVal);
        std::cout << "emdw things are done...\n\n\n\n\n";

        bool DEBUG = true;
        size_t maxIters = 10;
        size_t m = 7;

        //*********************************************************
        // Load In Data
        // *********************************************************
        // Specify CSV Path
        std::string inpPathCSV = "../../../data/synthetic/test.csv";

        // Load Data
        debugPrint(DEBUG,
                   "Loading Drought Indices From '" + inpPathCSV + "' ...");
        std::vector<std::vector<int>> observedAttributes;
        try {
            observedAttributes = readData(inpPathCSV);
        } catch (std::exception &e) {
            std::cerr << "ERROR: " << e.what() << '\n';
            return 1;
        }

        // Print Attributes
        if (DEBUG) print2DArray(observedAttributes);

        // modelSelection(observedAttributes, std::make_pair(3, 10),
        // maxIters, 10,
        //                "../../../data/synthetic/modelSelection.csv",
        //                DEBUG);

        // return 0;

        runModel(m, observedAttributes, maxIters, "../../../data/synthetic/",
                 DEBUG);

        return 0;  // tell the world that all is fine
    }  // try

    catch (char msg[]) {
        cerr << msg << endl;
    }  // catch

    // catch (char const* msg) {
    //   cerr << msg << endl;
    // } // catch

    catch (const string &msg) {
        cerr << msg << endl;
        throw;
    }  // catch

    catch (const exception &e) {
        cerr << "Unhandled exception: " << e.what() << endl;
        throw e;
    }  // catch

    catch (...) {
        cerr << "An unknown exception / error occurred\n";
        throw;
    }  // catch
}

void debugPrint(bool debug, std::string message) {
    if (debug) std::cout << message << '\n';
}

// ==================== Helper Functions ====================
/**
 * @brief
 *  - Splits input string by given delimiter
 * @param
 *  - `inputString`: Input String
 *  - `delimiter`: Character To Split By
 * @return
 *  - Vector of split elements
 */
std::vector<std::string> splitByCharacter(const std::string &inputString,
                                          const char &delimiter) {
    std::vector<std::string> outp;

    size_t start = 0;
    size_t end = 0;

    while ((end = inputString.find(delimiter, start)) != std::string::npos) {
        outp.push_back(inputString.substr(start, end - start));
        start = end + 1;
    }

    outp.push_back(inputString.substr(start));
    return outp;
}

/**
 * @brief
 *  - Lil Bit of ASCII Art here :)
 */
template <typename T>
void print2DArray(std::vector<std::vector<T>> inp, std::string_view arrayName) {
    std::pair<int, int> dims = std::make_pair(inp.size(), inp[0].size());
    std::cout << "Printing " << arrayName << "...\n";

    std::cout << "   ";
    for (size_t i = 0; i < dims.second; i++) {
        std::cout << i << "  ";
    }

    std::cout << '\n';
    for (size_t i = 0; i < dims.first; i++) {
        std::cout << i << " |";
        for (size_t j = 0; j < dims.second; j++) {
            std::cout << inp.at(i).at(j);

            if (j != dims.second - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "|\n";
    }
}

/**
 * @brief
 *  - Lil Bit of ASCII Art here :)
 */
template <typename T>
void printVector(std::vector<T> inp, std::string_view vectorName) {
    size_t vecSize = inp.size();
    std::cout << "Printing " << vectorName << "...\n";

    std::cout << "[";
    for (size_t i = 0; i < vecSize; i++) {
        std::cout << inp.at(i);
        if (i != vecSize - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

/**
 * @brief
 *  - Helper function used for randomly initialising params
 * @param
 *  - `vecSize`: Size of random output vector
 *  - `mean`: Mean of normal distr (default = 0.0)
 *  - `variance`: Variance of normal distr (default = 1.0)
 * @return
 *  - Vector of postive random values of type `T`
 */
template <typename T>
std::vector<T> createRandomVector(const size_t vecSize, const T mean,
                                  const T variance) {
    // Ensure given mean & variance is arithmetic
    if (!std::is_arithmetic<T>::value)
        throw std::invalid_argument(
            "Given type `T` is not one of [int, float, double]");

    std::vector<T> outp;
    // This our random seed source
    std::random_device rd{};
    // Create RNG
    std::mt19937 gen{rd()};
    // Create normal distr object we can sample from
    std::normal_distribution<T> randomNormal(mean, variance);

    for (size_t i = 0; i < vecSize; i++) {
        outp.push_back(std::abs(randomNormal(gen)));
    }
    return outp;
}

/**
 * @brief
 *  - Find Max of a given 2D array along a given axis
 *  - Recall Axis things:
 *      - if `axis=0` then we collapse the first dimension, meaning finding
 * max values over the columns
 *      - Example:
 *          std::vector<std::vector<int>> matrix = {{1, 2, 9},
 *                                                  {4, 8, 6},
 *                                                  {7, 5, 3}};
 *          auto col_max = findMaxAlongAxis(matrix, axis=0);
 *          -> Returns {7, 8, 9}
 */
template <typename T>
std::vector<T> findMaxAlongAxis(const std::vector<std::vector<T>> &inp,
                                size_t axis) {
    // Handle empty input
    if (inp.empty()) {
        return std::vector<T>();
    }

    if (axis == 0) {
        // Find max along columns (axis 0) - max of each column
        size_t num_cols = inp[0].size();
        std::vector<T> result(num_cols, std::numeric_limits<T>::lowest());

        for (const auto &row : inp) {
            // Check if all rows have the same number of columns
            if (row.size() != num_cols) {
                throw std::invalid_argument(
                    "All rows must have the same number of columns");
            }

            for (size_t j = 0; j < num_cols; ++j) {
                if (row[j] > result[j]) {
                    result[j] = row[j];
                }
            }
        }
        return result;
    } else if (axis == 1) {
        // Find max along rows (axis 1) - max of each row
        std::vector<T> result;
        result.reserve(inp.size());

        for (const auto &row : inp) {
            if (row.empty()) {
                result.push_back(std::numeric_limits<T>::lowest());
                continue;
            }

            T max_val = row[0];
            for (size_t j = 1; j < row.size(); ++j) {
                if (row[j] > max_val) {
                    max_val = row[j];
                }
            }
            result.push_back(max_val);
        }
        return result;
    } else {
        throw std::invalid_argument("Axis must be 0 or 1");
    }
}

/**
 * @brief
 *  - Prints Domain of Drought State RVs
 * @param
 *  - `debug`: Debug flag
 *  - `droughtStateDomain`: Domain of S_t
 */
void printDomainDroughtState(
    bool debug, const rcptr<std::vector<int>> &droughtStateDomain) {
    if (!debug) return;

    std::cout << "→ Drought State Domain: [";
    for (size_t s = 0; s < droughtStateDomain->size(); s++) {
        std::cout << droughtStateDomain->at(s);
        if (s != droughtStateDomain->size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    return;
}

/**
 * @brief
 *  - Prints Domain of Discerete Attribute RVs
 * @param
 *  - `debug`: Debug flag
 *  - `attributeRvDomains`: Domains of A_t^n
 */
void printDomainAttributeRVsDiscerete(
    bool debug,
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains) {
    if (!debug) return;

    for (size_t n = 0; n < attributeRvDomains.size(); n++) {
        std::cout << "→ A^" << n + 1 << " Domain: [";

        for (size_t t = 0; t < attributeRvDomains.at(n)->size(); t++) {
            std::cout << attributeRvDomains.at(n)->at(t);
            if (t != attributeRvDomains.at(n)->size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    return;
}

/**
 * @brief
 *  - Prints a factor without the trailing 10 lines
 *  - Also removes the "DiscreteTable_V0" at the beginning
 * @param
 *  - `factor`: Factor to be printed
 *  - `factorName`: Factor's name
 */
void printFactor(const Factor &factor, std::string_view factorName) {
    std::stringstream buffer;
    std::streambuf *old_cout = std::cout.rdbuf(buffer.rdbuf());

    std::cout << factor << '\n';
    std::cout.rdbuf(old_cout);

    std::string output = buffer.str();
    std::istringstream iss(output);
    std::string line;
    std::vector<std::string> cleaned_lines;

    // Process each line
    while (std::getline(iss, line)) {
        // Skip the DiscreteTable_V0 line and the trailing function lines
        if (line.find("DiscreteTable_V0") == std::string::npos &&
            line.find("DiscreteTable_") != 0) {
            cleaned_lines.push_back(line);
        }
    }

    // Print the cleaned output
    std::cout << "--------- Printing " << factorName << " ---------\n";
    for (const auto &clean_line : cleaned_lines) {
        std::cout << clean_line << '\n';
    }
    std::cout << "----------------------------------------\n\n";
}

// ==================== File Handling ====================
/**
 * @brief
 *  - Reads in Drought Inidice Attributes.
 *  - Must be in Form (A^1, A^2, A^3, ....)
 * @param
 *  -
 * @return
 *  - 2D Array (Number Attributes, Value At Each Time Step) → (N, T)
 */
std::vector<std::vector<int>> readData(std::string_view filePath,
                                       bool hasHeader) {
    std::ifstream fin;
    std::string line;
    std::vector<std::vector<int>> outp;

    fin.open(filePath);
    if (!fin) {
        throw std::runtime_error("Given `filePath = " + std::string(filePath) +
                                 "` could not be opened...");
    }

    // Skip a Line If There Is a Header
    if (hasHeader) std::getline(fin, line);

    while (std::getline(fin, line)) {
        std::vector<std::string> lineElements = splitByCharacter(line, ',');

        std::vector<int> timeStep;
        for (size_t i = 0; i < lineElements.size(); i++) {
            int value = std::stoi(lineElements.at(i));
            timeStep.push_back(value);
        }
        outp.push_back(timeStep);
    }

    fin.close();
    return outp;
}

/**
 * @brief
 *  - Processes and Saves Model Output As two CSV files
 *  - Ouptuts one for MPM Rule & One for Vertibi Algo
 * @param
 *  - `dirPath`: Where to save CSVs
 *  - `cg`: Output of LBU
 *  - `msgs`: Output of LBU
 *  - `droughtStateRvIDs`: IDs of all S_t
 *  -  `finalPriors`: Final Prior Probs From Model
 *  - `finalTrans`: Final Transition Probs From Model
 *  - `finalEmissions`: Final Emission Probs From Model
 *  - `observedAttributes`: Observed Data
 */
void processAndSaveModelOutput(
    std::string_view dirPath, ClusterGraph &cg,
    std::map<Idx2, rcptr<Factor>> &msgs,
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<double> &finalPriors,
    const std::vector<std::vector<double>> &finalTrans,
    const std::vector<std::vector<std::vector<double>>> &finalEmissions,
    const std::vector<std::vector<int>> &observedAttributes) {
    size_t T = droughtStateRvIDs.size();
    size_t N = observedAttributes.at(0).size();
    size_t m = finalPriors.size();

    // ==================== MPM Rule ====================
    std::vector<int> MPMrule;
    for (size_t t = 0; t < T; t++) {
        rcptr<Factor> qSt =
            queryLBU_CG(cg, msgs, {droughtStateRvIDs.at(t)})->normalize();

        // Find Max Confidence of p(S_t)
        double maxConf = 0.0;
        int maxVal = -1;
        for (int i = 1; i <= m; i++) {
            double currConf = qSt->potentialAt({droughtStateRvIDs.at(t)}, {i});
            if (currConf > maxConf) {
                maxConf = currConf;
                maxVal = i;
            }
        }

        // Add Max Value
        if (maxVal == -1)
            throw std::runtime_error(
                "Extracting Of Drought States Went Wrong...");

        MPMrule.push_back(maxVal);
    }

    // ==================== Vertibi ====================
    // First need p(\vec{A}_t | S_t = i)
    //  - Working in log-space, and will simply exec exp(log(p(...)))
    //      if I want to use it
    std::vector<std::vector<double>> log_pAgSi(T, std::vector<double>(m, 0.0));
    for (size_t t = 0; t < T; t++) {
        for (size_t i = 0; i < m; i++) {
            for (size_t n = 0; n < N; n++) {
                int obs = observedAttributes.at(t).at(n);
                double p = finalEmissions.at(n).at(obs - 1).at(i);
                log_pAgSi[t][i] += std::log(p);
            }
        }
    }

    // Then need the rest in log space
    std::vector<double> logPriors;
    for (const auto &elem : finalPriors) {
        logPriors.push_back(std::log(elem));
    }
    std::vector<std::vector<double>> logTrans;
    for (const auto &row : finalTrans) {
        std::vector<double> inpRow;
        for (const auto &elem : row) {
            inpRow.push_back(std::log(elem));
        }
        logTrans.push_back(inpRow);
    }

    auto [vertibiPath, vertibiLogProb] =
        viterbi(logPriors, logTrans, log_pAgSi);

    // ==================== Save Output ====================
    // MPM
    std::ofstream fout;
    fout.open(std::string(dirPath) + "mpm_rule_output.csv");
    if (!fout)
        throw std::invalid_argument("Could Not Access: `" +
                                    std::string(dirPath) + '`');

    fout << "St\n";
    for (size_t t = 0; t < T; t++) {
        // Populate CSV with value
        fout << MPMrule.at(t) << '\n';
    }
    fout.close();

    // Veterbi
    fout.open(std::string(dirPath) + "veterbi_output.csv");
    if (!fout)
        throw std::invalid_argument("Could Not Access: `" +
                                    std::string(dirPath) + '`');

    fout << "St,log_prob\n"
         << std::to_string(vertibiPath.at(0)) << ','
         << std::to_string(vertibiLogProb) << '\n';
    for (size_t t = 1; t < T; t++) {
        // Populate CSV with value
        fout << vertibiPath.at(t) << ",\n";
    }
    fout.close();
}

/**
 * @brief
 *  - Saves output of Model Seclection as a CSV
 *      - {Log-Likelihood, AIC, BIC} at different values of `m`
 * @param
 *  - `outputPathCSV`: Output path for where to save the CSV
 *  - `results`: Model Selection Output
 *      - 2D array of m vectors of size 3: {Log-Likelihood, AIC, BIC}
 *  - `mMin` Starting `m`
 */
void saveModelSelectionOutput(std::string_view outputPathCSV,
                              const std::vector<std::vector<double>> &results,
                              const size_t mMin) {
    std::ofstream fout;
    fout.open(outputPathCSV);
    if (!fout)
        throw std::invalid_argument("Given `filePath` could not be opened: " +
                                    std::string(outputPathCSV));

    debugPrint(debug, "Saving Output...");
    // Heading
    fout << "m,log_lik,aic,bic\n";
    for (size_t i = 0; i < results.size(); i++) {
        fout << std::to_string(mMin + i) << ','
             << std::to_string(results.at(i).at(0)) << ','
             << std::to_string(results.at(i).at(1)) << ','
             << std::to_string(results.at(i).at(2)) << '\n';
    }
    fout.close();
}

// ==================== Creating Params ====================
/**
 * @brief
 *  - Used for initialisation of prior parameters [π_1, π_2, ..., π_m]
 * @param
 *  - `m`: Cardinality of S_t
 *  - `mean`: Mean of normal distr (default = 0.0)
 *  - `variance`: Variance of normal distr (default = 1.0)
 * @return
 *  - Vector of random values of type `double`
 */
std::vector<double> createRandomPriors(const int m, const double mean,
                                       const double variance) {
    return createRandomVector(m, mean, variance);
}

/**
 * @brief
 *  - Used for initialisation of transition params P^1
 * @param
 *  - `m`: Cardinality of S_t
 *  - `mean`: Mean of normal distr (default = 0.0)
 *  - `variance`: Variance of normal distr (default = 1.0)
 * @return
 *  - 2D Array (mxm) of random values of type `double`
 */
std::vector<std::vector<double>> createRandomTransitionParams(
    const int m, const double mean, const double variance) {
    std::vector<std::vector<double>> outp;

    for (size_t i = 0; i < m; i++) {
        outp.push_back(createRandomVector(m, mean, variance));
    }
    return outp;
}

/**
 * @brief
 *  - Creates parameters for attribute potentials randomly according to
 * normal distribution
 *  - See tablet for DS explanation
 * @param
 *  - `attributeRvDomains`: Domains of Attribute RVs (A^n_t),
 *  - `m`: Number of values hidden drought state can take on
 *  - `mean` = 0.0: Mean of normal distribution
 *  - `variance` = 1.0: Variance of normal distribution
 * @return
 *  - Returns Parameters
 */
std::vector<std::vector<std::vector<double>>> createEmissionParams(
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains,
    const size_t m, const double variance, const double mean) {
    std::vector<std::vector<std::vector<double>>> emissionProbs;

    size_t N = attributeRvDomains.size();

    // This our random seed source
    std::random_device rd{};
    // Create RNG
    std::mt19937 gen{rd()};
    // Create normal distr object we can sample from
    std::normal_distribution<double> randomNormal(mean, std::sqrt(variance));
    // Create Bn
    for (int n = 0; n < N; n++) {
        size_t Cn = attributeRvDomains.at(n)->size();

        // Prepopulate with 0s, then cylce through each one with reference
        //  to alter them to random numbers
        std::vector<std::vector<double>> Bn(Cn, std::vector<double>(m, 0.0));
        for (auto &row : Bn) {
            for (auto &elem : row) {
                elem = std::abs(randomNormal(gen));
            }
        }

        // Now add this matrix to our vector of matrices
        emissionProbs.push_back(Bn);
    }
    return emissionProbs;
}

// ==================== Model Setup ====================
/**
 * @brief
 *  - Returns a vector of emdw RV IDs, incrementing from given starting
 * point
 * @param
 *  -
 *  - numNodes: Number of RV IDs to create
 *  - runningIdCount: Where to begin incrementing from (inclusive)
 * @return
 *  -
 */
std::vector<emdw::RVIdType> createDiscreteRvIds(int numNodes,
                                                uint &runningIdCount) {
    std::vector<emdw::RVIdType> nodeIds;
    for (int i = 0; i < numNodes; i++) {
        nodeIds.push_back(runningIdCount);
        runningIdCount += 1;
    }
    return nodeIds;
}

/**
 * @brief
 *  - This will create a `rcptr` of a vector of ints
 *  - These ints then increment to max C
 *  - Example with C = 3:
 *      output: [1,2,3]
 */
rcptr<std::vector<int>> createDiscreteRvDomain(size_t C) {
    rcptr<std::vector<int>> domain(new std::vector<int>());
    for (int i = 1; i <= C; ++i) {
        domain->push_back(i);
    }
    return domain;
}

/**
 * @brief
 *  - See tablet notes for explanation
 * @param
 *  - `droughtStateDomain`: Domain of S_t
 *  - `oldPriors`: priors / parameters
 *  - `S1id`: Id of RV S_1
 *  - `noiseVariance`: Amount of variacne to break uniform distr
 * @return
 *  - p(S_1)
 */
rcptr<Factor> createPriorS1Factor(
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<double> &oldPriors, const emdw::RVIdType &S1id,
    const double &noiseVariance) {
    size_t m = droughtStateDomain->size();

    // ==================== Lil Assurance Check ====================
    if (m != oldPriors.size()) {
        throw std::invalid_argument(
            "number prior S_1 estimates do not match domain of S_1...");
    }

    // ==================== Check If Close To Uniform ====================
    //  - Checks if all elements are nearly identical to first element
    //  - Rather crude....
    bool isUniform = false;
    double runningDiff = 0.0;
    for (int i = 1; i < m; i++) {
        runningDiff += std::abs(oldPriors[0] - oldPriors[i]);
    }
    if (runningDiff < std::pow(m, -2)) isUniform = true;

    // ===== Create noise variable to conditionally eradicate symmetry =====
    // This our random seed source
    std::random_device rd{};
    // Create RNG
    std::mt19937 gen{rd()};
    // Create normal distr object we can sample from
    std::normal_distribution<double> noise_rv(0, std::sqrt(noiseVariance));

    // ============== Dynamically Define Factor Potentials ==============
    std::map<std::vector<int>, FProb> sparseProbs;

    // Probs is taken from prior estimates
    for (int i = 0; i < m; i++) {
        // If uniform then add noise else add 0.0
        sparseProbs[{droughtStateDomain->at(i)}] =
            oldPriors.at(i) + (isUniform ? noise_rv(gen) : 0.0);
    }

    // ==================== Create P(S_1) ====================
    rcptr<Factor> factorS1 =
        uniqptr<DT>(new DT({S1id}, {droughtStateDomain}, defProb, sparseProbs));

    // Lets normalise the factor as well
    factorS1 = factorS1->normalize();

    return factorS1;
}

/**
 * @brief
 *  - See tablet notes for explanation
 * @param
 *  - `droughtStateDomain`: Domain of S_t RV
 *  - `transitionMatrix`: Input Probability Parameters
 *  - `droughtStateIds`: Ids for all S_t RVs
 * @return
 *  - Vector of Transition Factors
 *      - Example: [p(S_2 | S_1), p(S_3 | S_2), ..., p(S_T | S_{T-1})]
 */
std::vector<rcptr<Factor>> createTransitionFactors(
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<std::vector<double>> &transitionMatrix,
    const std::vector<emdw::RVIdType> &droughtStateIds) {
    size_t m = droughtStateDomain->size();
    size_t T = droughtStateIds.size();

    // ==================== Lil Assurance Check ====================
    if ((transitionMatrix.size() != m) ||
        (transitionMatrix.at(0).size() != m)) {
        throw std::invalid_argument(
            "Given `transitionMatrix` is not size (m, m)");
    }

    // ============== Dynamically Define Factor Potentials ==============
    std::map<std::vector<int>, FProb> sparseProbs;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            sparseProbs[{droughtStateDomain->at(i),
                         droughtStateDomain->at(j)}] =
                transitionMatrix.at(i).at(j);
        }
    }

    // ==================== Create P(S_{t+1} | S_t) ====================
    // - See tablet for proof that we have T-1 P(S_{t+1} | S_t) factors
    std::vector<rcptr<Factor>> transitionFactors;

    // Create p(S_2 | S_1)
    transitionFactors.push_back(uniqptr<DT>(new DT(
        {droughtStateIds.at(0), droughtStateIds.at(1)},
        {droughtStateDomain, droughtStateDomain}, defProb, sparseProbs)));

    // Simply copy for all other p(S_3 | S_2), ..., p(S_T | S_{T-1})
    for (size_t t = 0; t < T - 1; t++) {
        if (t != 0) {
            transitionFactors.push_back(
                uniqptr<Factor>(transitionFactors.at(0)->copy(
                    {droughtStateIds.at(t), droughtStateIds.at(t + 1)})));
        }

        // Lets normalise the factor as well
        transitionFactors[t] = transitionFactors.at(t)->normalize();
    }

    return transitionFactors;
}

/**
 * @brief
 *  - See tablet notes for explanation
 * @param
 *  - `droughtStateRvIDs`: IDs Associated With S_t RVs
 *  - `attributeRvIds`: IDs Associated With A^n_t RVs
 *  - `droughtStateDomain`: Domain For S_t
 *  - `attributeRvDomains`: Domain For A^n
 *  - `emissionProbs`: Input parameters for potentials of factors
 * @return
 *  - 2D Array of Emission Factors (See Tablet For DS)
 */
std::vector<std::vector<rcptr<Factor>>> createEmissionFactors(
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains,
    const std::vector<std::vector<std::vector<double>>> &emissionProbs) {
    size_t m = droughtStateDomain->size();
    size_t N = attributeRvDomains.size();
    size_t T = droughtStateRvIDs.size();
    std::vector<size_t> CnVals(N, 0);

    // ================= Lil Assurance Check (Input Params)
    // =================
    for (size_t n = 0; n < N; n++) {
        CnVals[n] = attributeRvDomains.at(n)->size();

        bool dim1Correct = (emissionProbs.size() == N);
        bool dim2Correct = (emissionProbs.at(n).size() == CnVals.at(n));
        bool dim3Correct = (emissionProbs.at(n).at(0).size() == m);

        if (!dim1Correct || !dim2Correct || !dim3Correct) {
            throw std::invalid_argument(
                "Given `emissionProbs` is not the correct shape. Must be "
                "(N, " +
                std::to_string(CnVals.at(n)) + ", m)\nRather, it is: (" +
                std::to_string(emissionProbs.size()) + ", " +
                std::to_string(emissionProbs.at(0).size()) + ", " +
                std::to_string(emissionProbs.at(0).at(0).size()) + ")");
        }
    }
    // ==================== Create Factors for t=1 ====================
    std::vector<rcptr<Factor>> originalEmissionFactor_t1;

    // ==================== Making N of these guys... ====================

    for (size_t n = 0; n < N; n++) {
        // ============== Dynamically Define Factor Potentials
        // ==============
        std::map<std::vector<int>, FProb> sparseProbs;

        // Cycle Through values of A^n
        for (size_t i = 0; i < CnVals.at(n); i++) {
            // Cycle Through values of S_t
            for (size_t j = 0; j < m; j++) {
                sparseProbs[{attributeRvDomains.at(n)->at(i),
                             droughtStateDomain->at(j)}] =
                    emissionProbs.at(n).at(i).at(j);
            }
        }

        // ==================== Create P(A^n_1 | S_1) ====================
        // - Only thing varying here is n
        // - Therefore the S_1 id stays contant here.
        // - Similarly, the index of attributeRvIds at t=1 stays constant
        originalEmissionFactor_t1.push_back(uniqptr<DT>(
            new DT({attributeRvIds.at(0).at(n), droughtStateRvIDs.at(0)},
                   {attributeRvDomains.at(n), droughtStateDomain}, defProb,
                   sparseProbs)));

        // Normalise the guy as well
        originalEmissionFactor_t1[n] =
            originalEmissionFactor_t1.at(n)->normalize();
    }

    // ================ Now we expand this vector down T times
    // ================
    std::vector<std::vector<rcptr<Factor>>> emissionFactors;

    // Again see tablet that our output is TxN
    for (size_t t = 0; t < T; t++) {
        std::vector<rcptr<Factor>> factorVectorToInsert;
        for (size_t n = 0; n < N; n++) {
            factorVectorToInsert.push_back(
                uniqptr<Factor>(originalEmissionFactor_t1.at(n)->copy(
                    {attributeRvIds.at(t).at(n), droughtStateRvIDs.at(t)})));
        }
        emissionFactors.push_back(factorVectorToInsert);
    }

    return emissionFactors;
}

// ==================== Validation Functions ====================
/**
 * @brief
 *  - Function to validate if RV ID generation was correct
 *  - Will simply cycle through each given data structure and see if they
 * are incrementing
 * @param
 *  -
 * @return
 *  - True (if Correct) or False (If Incorrect)
 */
bool validateRvIds(const bool &debug,
                   const std::vector<emdw::RVIdType> &droughtStateIds,
                   const std::vector<std::vector<emdw::RVIdType>> &attributeIds,
                   const std::vector<std::vector<int>> &observedAttributes) {
    int lastItem = -1;

    // Cylce Through hidden drought states
    for (const emdw::RVIdType &droughtID : droughtStateIds) {
        if (lastItem != droughtID - 1) return false;
        lastItem += 1;
    }

    debugPrint(debug, "Hidden Drought RVs Are Correctly Initialised");

    // ensure that `attributeIds` & `observedAttributes` are the same size
    if ((observedAttributes.size() != attributeIds.size()) ||
        (observedAttributes.at(0).size() != attributeIds.at(0).size())) {
        return false;
    }
    // Cylce Through attribute RVs (Cycling the same as they were
    // created...)
    size_t T = attributeIds.size();
    size_t N = attributeIds.at(0).size();
    for (size_t t = 0; t < T; t++) {
        for (size_t n = 0; n < N; n++) {
            if (lastItem != attributeIds.at(t).at(n) - 1) return false;
            lastItem += 1;
        }
    }
    debugPrint(debug, "Attribute RVs Are Correctly Initialised");
    return true;
}

// ==================== Model Inference ====================

/**
 * @brief
 *  - Performs LBU with the LTRIP Algo and for Discrete A^n_t RVs
 * @param
 *  -  `pS1`: Factor For → p(S_1)
 *  - `transitionFactors`: Transition Factors → p(S_{t+1} | S_t)
 *  - `emissionFactors`: Emission Factors → p(A^n_t | S_t)
 *  - `droughtStateRvIDs`: IDs for S_t RVs
 *  - `attributeRvIds`: IDs for A^n_t RVs
 *  - `observedAttributes`: Observed Data
 * @return
 *  - Package of two variables that is needed for querying
 */
std::pair<ClusterGraph, std::map<Idx2, rcptr<Factor>>> performLBU_LTRIP(
    const rcptr<Factor> &pS1,
    const std::vector<rcptr<Factor>> &transitionFactors,
    const std::vector<std::vector<rcptr<Factor>>> &emissionFactors,
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const std::vector<std::vector<int>> &observedAttributes) {
    // ==================== Init Things ====================
    size_t T = droughtStateRvIDs.size();
    size_t N = attributeRvIds.at(0).size();

    // ================ Put All Factors In A Single Vector ================
    // - Order doesn't really matter

    // Copy In p(S_1) now already
    std::vector<rcptr<Factor>> factorsVector = {
        uniqptr<Factor>(pS1->copy({droughtStateRvIDs.at(0)}))};

    // Copy In Transition Factors (of size T-1)
    for (size_t t = 0; t < T - 1; t++) {
        factorsVector.push_back(uniqptr<Factor>(transitionFactors.at(t)->copy(
            {droughtStateRvIDs.at(t), droughtStateRvIDs.at(t + 1)})));
    }

    // Insert Emission Factors (of size (T, N))
    for (size_t t = 0; t < T; t++) {
        for (size_t n = 0; n < N; n++) {
            factorsVector.push_back(
                uniqptr<Factor>(emissionFactors.at(t).at(n)->copy(
                    {attributeRvIds.at(t).at(n), droughtStateRvIDs.at(t)})));
        }
    }

    // ================ Assign All Observed Variables ================
    // The result is a `std::map` where the key is the ID of the observed RV
    //  while the value is the observed data
    std::map<emdw::RVIdType, AnyType> observedData;

    for (size_t t = 0; t < T; t++) {
        for (size_t n = 0; n < N; n++) {
            observedData[attributeRvIds.at(t).at(n)] =
                observedAttributes.at(t).at(n);
        }
    }

    // ==================== The Rest Is Just Copied ====================

    // Create Clustergraph
    ClusterGraph cg(ClusterGraph::LTRIP, factorsVector, observedData);
    /* std::cout << cg << std::endl; */

    // export the graph to graphviz .dot format
    // cg.exportToGraphViz("jtree");

    // Now Calibrate the graph
    std::map<Idx2, rcptr<Factor>> msgs;
    MessageQueue msgQ;

    msgs.clear();
    msgQ.clear();
    unsigned nMsgs = loopyBU_CG(cg, msgs, msgQ);
    std::cout << "Sent " << nMsgs << " messages before convergence\n";

    return std::make_pair(cg, msgs);
}

/**
 * @brief
 *  - Creates model & runs EM Algorithm
 *  - If valid `outputPathCSV` is given it extracts posterior chain
 *  - if invalud path, will then return the final paramaters (for model
 *  seclection)
 * @param
 *  - `m`: Cardinality of S_t
 *  - `observedAttributes`: Data
 *  - `maxIters`: Maximum number of EM iters
 *  - `outputDir`: Which Directory to save output
 *      - if != "None" will save else will skip
 *      - Must have have trailing `/`
 * output chain)
 *  - `debug`: flag that triggers debug printing
 * @return
 *  - A tuple: {Priors, Transition Matrix, Emission Probabilities}
 */
std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<std::vector<std::vector<double>>>>
runModel(const size_t m,
         const std::vector<std::vector<int>> &observedAttributes,
         const size_t maxIters, std::string_view outputDir, const bool debug) {
    //*********************************************************
    // Specify Parameters
    // *********************************************************
    // NOTE: Not sure if conditionally breaking symmetry is necessary...
    //  Can't hurt I suppose.
    float priorNoise = 0.005;

    // Create Prior Params
    std::vector<double> oldPriors = createRandomPriors(m, 0.0, 1.0);
    // Create Transition Params
    std::vector<std::vector<double>> oldTransitionMatrix =
        createRandomTransitionParams(m, 0.0, 1.0);

    // Ensure sizes
    assert(m == oldPriors.size());
    assert((m == oldTransitionMatrix.size()) &&
           (m == oldTransitionMatrix.at(0).size()));

    // ============ Populate Useful Variables Here, Now  ============
    size_t T = observedAttributes.size();
    size_t N = observedAttributes.at(0).size();
    debugPrint(debug, "# Time Steps → T = " + std::to_string(T));
    debugPrint(debug, "# Attribute RVs → N = " + std::to_string(N));

    debugPrint(debug, "✓ Success!\n");

    //*********************************************************
    // Generate RV IDs
    // *********************************************************

    debugPrint(debug, "Creating RV IDs...");

    // Starting ID
    uint rvIdentity = 0;

    // We will have T S_t RVs
    std::vector<emdw::RVIdType> droughtStateRvIDs =
        createDiscreteRvIds(T, rvIdentity);

    // Will access A_t^n IDs as a 2D matrix
    std::vector<std::vector<emdw::RVIdType>> attributeRvIds;
    for (size_t t = 0; t < T; t++) {
        std::vector<emdw::RVIdType> singleAttributeRvIds =
            createDiscreteRvIds(N, rvIdentity);
        attributeRvIds.push_back(singleAttributeRvIds);
    }

    // Check that all values are unique and of type `emdw::RVIdType`
    if (!validateRvIds(debug, droughtStateRvIDs, attributeRvIds,
                       observedAttributes))
        throw std::runtime_error("ERROR: Generation Of RV IDs Failed...\n");

    debugPrint(debug, "✓ Success!\n");

    //*********************************************************
    // Create Domain Of RVs
    // *********************************************************
    //  - We need a `rcptr` of a vector of ints
    //  - This vector of ints will represent all the possible values the
    //  RV can take on.
    //  - This is quite simple but still out sourcing to a sperate
    //  function

    debugPrint(debug, "Creating Domains Of The RVs Of The Model...");

    // Hidden Drought State is simply `[1, 2, ..., m]`
    rcptr<std::vector<int>> droughtStateDomain = createDiscreteRvDomain(m);

    // Attribute RVs
    std::vector<rcptr<std::vector<int>>> attributeRvDomains;

    // Along columns, thus `maxVals.size()` == N
    std::vector<int> maxVals = findMaxAlongAxis(observedAttributes, 0);

    for (size_t n = 0; n < N; n++) {
        attributeRvDomains.push_back(createDiscreteRvDomain(maxVals.at(n)));
    }

    printDomainDroughtState(debug, droughtStateDomain);
    printDomainAttributeRVsDiscerete(debug, attributeRvDomains);

    debugPrint(debug, "✓ Success!\n");

    //*********************************************************
    // Define Factors Of Model
    // *********************************************************
    //  - Here we get the factors of the model, ie. the probability
    //  tables
    //      and things

    debugPrint(debug, "Creating Factors Of The Model...");

    // ==================== Priors → p(S_1) ====================
    rcptr<Factor> pS1 = createPriorS1Factor(
        droughtStateDomain, oldPriors, droughtStateRvIDs.at(0), priorNoise);

    if (debug) printFactor(*pS1, "p(S_1)");

    // ================= Transition → p(S_{t+1} | S_t) =================
    std::vector<rcptr<Factor>> transitionFactors = createTransitionFactors(
        droughtStateDomain, oldTransitionMatrix, droughtStateRvIDs);

    if (debug) printFactor(*transitionFactors.at(0), "p(S_2 | S_1)");

    // Change this flag for very comprehensive test
    if (false) {
        if (debug) printFactor(*transitionFactors.at(1), "p(S_3 | S_2)");
        if (debug)
            printFactor(*transitionFactors.at(transitionFactors.size() - 1),
                        "p(S_" + std::to_string(T) + " | S_" +
                            std::to_string(T - 1) + ")");
        print2DArray(oldTransitionMatrix);
    }

    // ================= Emission → p(A_t^n | S_t) =================
    // Create Parameters
    std::vector<std::vector<std::vector<double>>> oldEmissionProbs =
        createEmissionParams(attributeRvDomains, m);

    assert(oldEmissionProbs.size() == N);

    if (debug) {
        for (int n = 0; n < N; n++) {
            print2DArray(oldEmissionProbs.at(n),
                         "Params For A^" + std::to_string(n + 1));
            std::cout << '\n';
        }
    }

    // Create Factors
    std::vector<std::vector<rcptr<Factor>>> emissionFactors =
        createEmissionFactors(droughtStateRvIDs, attributeRvIds,
                              droughtStateDomain, attributeRvDomains,
                              oldEmissionProbs);

    if (debug) printFactor(*emissionFactors.at(0).at(1), "p(A^2_1 | S_1)");

    debugPrint(debug, "✓ Success!\n");
    //*********************************************************
    // EM ALGORITHM!!!!
    // *********************************************************

    debugPrint(debug, "Performing EM Algorithm...");

    // ==================== Initialise Loop Params ====================
    size_t numIter = 1;
    // Timer
    auto startEM = std::chrono::high_resolution_clock::now();
    auto endEM = std::chrono::high_resolution_clock::now();
    while (1) {
        // ============= Obtain p(H|D, Theta) (E-Step) ============
        std::pair<ClusterGraph, std::map<Idx2, rcptr<Factor>>> outp =
            performLBU_LTRIP(pS1, transitionFactors, emissionFactors,
                             droughtStateRvIDs, attributeRvIds,
                             observedAttributes);
        auto &[cg, msgs] = outp;

        // ================ Parameter Update (M-Step) ================

        // Cache marginals to avoid redundant queries
        std::vector<rcptr<Factor>> marginalCache(T);
        for (size_t t = 0; t < T; t++) {
            marginalCache[t] =
                queryLBU_CG(cg, msgs, {droughtStateRvIDs.at(t)})->normalize();
        }

        // Cache pairwise marginals for transitions
        std::vector<rcptr<Factor>> pairwiseCache(T - 1);
        for (size_t t = 0; t < T - 1; t++) {
            pairwiseCache[t] = queryLBU_CG(cg, msgs,
                                           {droughtStateRvIDs.at(t),
                                            droughtStateRvIDs.at(t + 1)})
                                   ->normalize();
        }

        // 1) Priors (π_j)
        std::vector<double> newPriors;
        for (size_t i = 1; i <= m; i++) {
            newPriors.push_back(marginalCache[0]->potentialAt({0}, {int(i)}));
        }

        if (newPriors.size() != oldPriors.size())
            throw std::runtime_error("Update Rule For Priors Failed...");

        // 2) Transition Probs (a_{i,j})
        std::vector<std::vector<double>> newTransitionMatrix;

        // Pre-compute denominators for all states
        //  - Note: stateCounts[0] is left at 0.0 and is never accessed
        std::vector<double> stateCounts(m + 1, 0.0);
        for (size_t t = 0; t < T - 1; t++) {
            for (size_t i = 1; i <= m; i++) {
                stateCounts[i] += marginalCache[t]->potentialAt(
                    {droughtStateRvIDs.at(t)}, {int(i)});
            }
        }

        // Numerator now
        for (size_t i = 1; i <= m; i++) {
            std::vector<double> newRow;

            for (size_t j = 1; j <= m; j++) {
                double runningNumerator = 0.0;
                for (size_t t = 0; t < T - 1; t++) {
                    runningNumerator += pairwiseCache[t]->potentialAt(
                        {droughtStateRvIDs.at(t), droughtStateRvIDs.at(t + 1)},
                        {int(i), int(j)});
                }
                newRow.push_back(runningNumerator / stateCounts[i]);
            }
            newTransitionMatrix.push_back(std::move(newRow));
        }

        // Sanity Check
        if ((newTransitionMatrix.size() != oldTransitionMatrix.size()) ||
            (newTransitionMatrix.at(0).size() !=
             oldTransitionMatrix.at(0).size()))
            throw std::runtime_error(
                "Update Rule For Transition Probs Failed...");

        // 3) Emission Probs (b_i^(n)(j))
        std::vector<std::vector<std::vector<double>>> newEmissionProbs;

        // Pre-compute state counts across all time steps (Denominator)
        //  - Note: totalStateCounts[0] is left at 0.0 and is never accessed
        std::vector<double> totalStateCounts(m + 1, 0.0);
        for (size_t t = 0; t < T; t++) {
            for (size_t i = 1; i <= m; i++) {
                totalStateCounts[i] += marginalCache[t]->potentialAt(
                    {droughtStateRvIDs.at(t)}, {int(i)});
            }
        }

        // Numerator now
        for (size_t n = 0; n < N; n++) {
            size_t Cn = attributeRvDomains.at(n)->size();

            // Size of B_n: (C_n, m)
            std::vector<std::vector<double>> inpMatrix(
                Cn, std::vector<double>(m, 0.0));

            // Count state-observation co-occurrences
            std::vector<std::vector<double>> stateObsCounts(
                m + 1, std::vector<double>(Cn + 1, 0.0));

            for (size_t t = 0; t < T; t++) {
                int observedValue = observedAttributes.at(t).at(n);
                for (size_t i = 1; i <= m; i++) {
                    stateObsCounts[i][observedValue] +=
                        marginalCache[t]->potentialAt({droughtStateRvIDs.at(t)},
                                                      {int(i)});
                }
            }

            // Compute emission probabilities
            for (size_t i = 1; i <= m; i++) {
                for (size_t j = 1; j <= Cn; j++) {
                    inpMatrix[j - 1][i - 1] =
                        stateObsCounts[i][j] / totalStateCounts[i];
                }
            }

            // `std::move()` transfers ownership to `newEmissionProbs`
            //  instead of copying the matrix over
            newEmissionProbs.push_back(std::move(inpMatrix));
        }

        // Sanity Check
        if ((newEmissionProbs.size() != oldEmissionProbs.size()) ||
            (newEmissionProbs.size() != N))
            throw std::runtime_error(
                "Update Rule For Emission Probs Failed...");
        for (size_t n = 0; n < N; n++) {
            size_t Cn = attributeRvDomains.at(n)->size();

            if (newEmissionProbs.at(n).size() !=
                    oldEmissionProbs.at(n).size() ||
                (newEmissionProbs.at(n).size() != Cn))
                throw std::runtime_error(
                    "Update Rule For Emission Probs Failed...");

            if ((newEmissionProbs.at(n).at(0).size() !=
                 oldEmissionProbs.at(n).at(0).size()) ||
                (newEmissionProbs.at(n).at(0).size() != m))
                throw std::runtime_error(
                    "Update Rule For Emission Probs Failed...");
        }

        // ================== Print Loop Information ==================

        if (debug) {
            std::cout << "Iteration " << numIter << '\n';
            printVector(oldPriors, "Old Priors");
            printVector(newPriors, "New Priors");

            print2DArray(oldTransitionMatrix, "Old Transition Matrix");
            print2DArray(newTransitionMatrix, "New Transition Matrix");

            // TODO: Make a print function for emission probabilities
            std::cout << "Not printing emission probs...\n";

            std::cout << "------------------------\n\n";
        }

        // ============= Update Old Params For Next Loop
        oldPriors = newPriors;
        oldTransitionMatrix = newTransitionMatrix;
        oldEmissionProbs = newEmissionProbs;

        // ============= Iterate Loop & Exit Condition =============
        numIter += 1;
        if (numIter > maxIters) {
            // End Timer
            endEM = std::chrono::high_resolution_clock::now();

            // This is the condition for saving output (AIC, BIC, Log L)
            if (outputDir != "None") {
                processAndSaveModelOutput(
                    outputDir, cg, msgs, droughtStateRvIDs, oldPriors,
                    oldTransitionMatrix, oldEmissionProbs, observedAttributes);
            }
            break;
        }

        // ============= Create New Model (New Factors) =============
        // Priors
        pS1 = createPriorS1Factor(droughtStateDomain, oldPriors,
                                  droughtStateRvIDs.at(0), priorNoise);
        // Transition
        transitionFactors = createTransitionFactors(
            droughtStateDomain, oldTransitionMatrix, droughtStateRvIDs);

        // Emission
        emissionFactors = createEmissionFactors(
            droughtStateRvIDs, attributeRvIds, droughtStateDomain,
            attributeRvDomains, oldEmissionProbs);
    }

    // ==================== End Timer ====================
    std::chrono::duration<double> elapsedEM = endEM - startEM;

    // ==================== Display Info ====================
    std::cout << '\n'
              << numIter << " Iterations of EM algorithm completed in : "
              << elapsedEM.count() << " seconds\n\n";

    debugPrint(debug, "✓ Success");

    // Now need to return the params
    //  - This will not be used when doing final inference with a set `m`
    //  - However, for model selection, we need to calculate likelihood
    //  which
    //      simply requires the final parameters
    return {oldPriors, oldTransitionMatrix, oldEmissionProbs};
}

/**
 * @brief
 *  - Sweeps across given range for `m` and records the log-likelihood, AIC
 * & BIC for each `m`
 *  - These values then get saved as a csv to
 * @param
 *  - `observedAttributes`: Data
 *  - `mRange`: should contain the range of `m` values to sweep through:
 *      - [min(m), max(m)] inclusive
 *  - `maxEMIters`: Maximum number of EM iters
 *  - `numRestartIters`: Number of random restarts per value of `m`
 *  - `outputPathCSV`: Save path for stats
 *  - `debug`: flag that triggers debug printing
 */
void modelSelection(const std::vector<std::vector<int>> &observedAttributes,
                    const std::pair<size_t, size_t> &mRange,
                    const size_t maxEMIters, const size_t numRestartIters,
                    std::string_view outputPathCSV, const bool debug) {
    size_t T = observedAttributes.size();
    size_t N = observedAttributes.at(0).size();

    std::vector<size_t> CnVals(N, 0);

    // Holding results in 2d array where each row is {loglik, AIC, BIC}. And
    // of
    //  course, we have `mRange.second - mRange.first` of these rows. So
    //  size of array is (mRange.second - mRange.first, 3)
    std::vector<std::vector<double>> results;

    for (size_t m = mRange.first; m <= mRange.second; m++) {
        // ==================== Many Restarts ====================
        double maxLogLik = -std::numeric_limits<double>::max();

        for (size_t r = 0; r < numRestartIters; r++) {
            debugPrint(debug, "=============== m = " + std::to_string(m) +
                                  " | r =" + std::to_string(r + 1) +
                                  " ===============");
            auto [priors, transitionProbs, emissionProbs] =
                runModel(m, observedAttributes, maxEMIters, "None", false);

            // ================== Calculate Log-Likelihood
            // ================== p(\vec{A}_t | S_t = i)
            //  - Working in log-space, and will simply exec
            //  exp(log(p(...)))
            //      if I want to use it
            std::vector<std::vector<double>> log_pAgSi(
                T, std::vector<double>(m, 0.0));
            for (size_t t = 0; t < T; t++) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t n = 0; n < N; n++) {
                        // Here we populating CnVals but only on very first
                        // iter
                        if (m == mRange.first && r == 0) {
                            CnVals[n] = emissionProbs.at(n).size();
                        }
                        int obs = observedAttributes.at(t).at(n);
                        double p = emissionProbs.at(n).at(obs - 1).at(i);
                        log_pAgSi[t][i] += std::log(p);
                    }
                }
            }

            // Forward Algorithm With Scaling
            std::vector<std::vector<double>> alpha(T,
                                                   std::vector<double>(m, 0.0));
            std::vector<double> c(T, 0.0);  // scaling constants

            // compute alpha_1
            for (size_t i = 0; i < m; i++) {
                alpha[0][i] = priors.at(i) * std::exp(log_pAgSi[0][i]);
                c[0] += alpha[0][i];
            }
            // normalize
            for (size_t i = 0; i < m; i++) alpha[0][i] /= c[0];

            // recursion
            for (size_t t = 1; t < T; t++) {
                for (size_t j = 0; j < m; j++) {
                    double sum_prev = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        sum_prev +=
                            alpha.at(t - 1).at(i) * transitionProbs.at(i).at(j);
                    }
                    alpha[t][j] = sum_prev * exp(log_pAgSi.at(t).at(j));
                }
                // scale
                for (size_t j = 0; j < m; j++) c[t] += alpha.at(t).at(j);
                for (size_t j = 0; j < m; j++) alpha[t][j] /= c.at(t);
            }

            // log-likelihood = - sum(log(c[t]))
            double logLik = 0.0;
            for (size_t t = 0; t < T; t++) logLik += std::log(c.at(t));

            debugPrint(debug, "Log-Likelihood = " + std::to_string(logLik));
            if (logLik > maxLogLik) {
                maxLogLik = logLik;
                debugPrint(debug, "Max Log Likelihood Updated");
            }
            debugPrint(debug, "\n");
        }

        // Number of free parameters
        size_t tmpSum = 0.0;
        for (size_t n = 0; n < N; n++) {
            tmpSum += CnVals.at(n) - 1;
        }
        size_t p = m * m - 1 + m * tmpSum;

        // Number of Data Points
        size_t k = T;

        double AIC = -2 * maxLogLik + 2 * p;
        double BIC = -2 * maxLogLik + p * std::log(k);

        // Save Results
        results.push_back({maxLogLik, AIC, BIC});
    }

    // Now to save the results :)
    saveModelSelectionOutput(outputPathCSV, results, mRange.first);
    debugPrint(debug, "✓ Success");
    return;
}

/**
 * @brief
 *  - Runs the Viterbi algorithm to find the most probable hidden state
 *      sequence.
 *  - Everything is done in log-space to prevent numerical underflow.
 * @param
 *  - `logPriors`: log of our final prior probabilities
 *  - `logTrans`: log of our final transition matrix
 *  - `logEmit`: log P(A_t | S_t = j), factorisation of across all N attributes
 * @return
 *  - A pair where
 *      - first: vector of length T, giving the most probable state at each
 *                  time.
 *      - second: the log-probability of the Viterbi path.
 */
std::pair<std::vector<int>, double> viterbi(
    const std::vector<double> &logPriors,
    const std::vector<std::vector<double>> &logTrans,
    const std::vector<std::vector<double>> &logEmit) {
    size_t T = logEmit.size();
    size_t m = logPriors.size();

    // Dimension checks
    if (logTrans.size() != m)
        throw std::invalid_argument("Transition matrix must have m rows.");
    for (size_t i = 0; i < m; i++) {
        if (logTrans[i].size() != m)
            throw std::invalid_argument("Transition matrix must be m x m.");
    }
    for (size_t t = 0; t < T; t++) {
        if (logEmit[t].size() != m)
            throw std::invalid_argument("Emission matrix must be T x m.");
    }

    // DP tables
    std::vector<std::vector<double>> delta(
        T, std::vector<double>(m, -std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> psi(T, std::vector<int>(m, -1));

    // Initialization
    for (size_t j = 0; j < m; j++) {
        delta[0][j] = logPriors[j] + logEmit[0][j];
    }

    // Recursion
    for (size_t t = 1; t < T; t++) {
        for (size_t j = 0; j < m; j++) {
            double max_val = -std::numeric_limits<double>::infinity();
            int argmax_i = -1;

            for (size_t i = 0; i < m; i++) {
                double val = delta[t - 1][i] + logTrans[i][j];
                if (val > max_val) {
                    max_val = val;
                    argmax_i = (int)i;
                }
            }
            delta[t][j] = max_val + logEmit[t][j];
            psi[t][j] = argmax_i;
        }
    }

    // Termination
    double best_log_prob = -std::numeric_limits<double>::infinity();
    int best_last_state = -1;
    for (size_t j = 0; j < m; j++) {
        if (delta[T - 1][j] > best_log_prob) {
            best_log_prob = delta[T - 1][j];
            best_last_state = (int)j;
        }
    }

    // Backtrace
    std::vector<int> best_path(T);
    best_path[T - 1] = best_last_state;
    for (int t = (int)T - 1; t > 0; t--) {
        best_path[t - 1] = psi[t][best_path[t]];
    }

    return {best_path, best_log_prob};
}
