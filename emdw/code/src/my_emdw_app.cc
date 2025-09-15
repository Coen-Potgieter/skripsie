// patrec headers
#include "prlite_logging.hpp"  // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "clustergraph.hpp"
#include "discretetable.hpp"
#include "emdw.hpp"
#include "lbu2_cg.hpp"  // ????
#include "lbu_cg.hpp"
#include "messagequeue.hpp"

// standard headers
#include <cassert>  // For assertions like python's `assert(condition)` functionality
#include <cctype>    // toupper
#include <iostream>  // cout, endl, flush, cin, cerr
#include <limits>    // Not Sure
#include <map>       // Sparse Probabilities for Factor Creation
#include <memory>    // Not Sure
#include <string>    // string
#include <vector>    // Everything

// Not using this but is being used by initialisation code
using namespace std;
using namespace emdw;

// Some Variables That Will Be Used Throughtout Various Functiosn
typedef DiscreteTable<int> DT;
double defProb = 0.0;

// ==================== Importing Data ====================
std::vector<std::vector<float>> readCSV(std::string_view filePath,
                                        bool hasHeader = true);
std::vector<std::string> splitByCharacter(const std::string &inputString,
                                          const char &delimiter);

// ==================== Printing Functions ====================
void debugPrint(bool debug, std::string message);
std::vector<emdw::RVIdType> createDiscreteRvIds(int numNodes,
                                                uint &runningIdCount);
void printDomainDroughtState(bool debug,
                             const rcptr<std::vector<int>> &droughtStateDomain);
void printDomainAttributeRVsDiscerete(
    bool debug, const std::vector<rcptr<std::vector<int>>> &attributeRvDomains);
void printFactor(const Factor &factor, std::string_view factorName);
template <typename T>
void print2DArray(std::vector<std::vector<T>> inp,
                  std::string_view arrayName = "2D Array");
template <typename T>
void printVector(std::vector<T> inp,
                 std::string_view vectorName = "Some Vector");
// ==================== Model Setup ====================
rcptr<std::vector<int>> createDiscreteRvDomain(size_t C);
rcptr<Factor> createPriorS1Factor(
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<double> &oldPriors, const emdw::RVIdType &S1id,
    const double &noiseVariance);
std::vector<rcptr<Factor>> createTransitionFactors(
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<std::vector<double>> &transitionMatrix,
    const std::vector<emdw::RVIdType> &droughtStateIds);
std::vector<std::vector<std::vector<double>>> createEmissionParamsDiscrete(
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains,
    const size_t m, const double variance = 1.0, const double mean = 0);
std::vector<std::vector<rcptr<Factor>>> createEmissionFactorsDiscrete(
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const rcptr<std::vector<int>> &droughtStateDomain,
    const std::vector<rcptr<std::vector<int>>> &attributeRvDomains,
    const std::vector<std::vector<std::vector<double>>> &emissionProbs);

// ==================== Helper Functions ====================
void saveModelOutput(std::string_view filePath, ClusterGraph &cg,
                     std::map<Idx2, rcptr<Factor>> &msgs, const int m,
                     const std::vector<emdw::RVIdType> &droughtStateRvIDs);
template <typename T>
std::vector<T> findMaxAlongAxis(const std::vector<std::vector<T>> &inp,
                                size_t axis = 0);

// ==================== Validation Functions ====================
bool validateRvIds(const bool &debug,
                   const std::vector<emdw::RVIdType> &droughtStateIds,
                   const std::vector<std::vector<emdw::RVIdType>> &attributeIds,
                   const std::vector<std::vector<float>> &observedAttributes);

// ==================== Model Inference ====================
std::pair<ClusterGraph, std::map<Idx2, rcptr<Factor>>>
performLBU_LTRIP_discrete(
    const rcptr<Factor> &pS1,
    const std::vector<rcptr<Factor>> &transitionFactors,
    const std::vector<std::vector<rcptr<Factor>>> &emissionFactors,
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const std::vector<std::vector<float>> &observedAttributes);

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

        //*********************************************************
        // Specify Parameters
        // *********************************************************

        bool DEBUG = true;
        size_t maxIters = 10;
        size_t m = 7;

        // Noise to conditionally break symmetry of the model
        float priorNoise = 0.005;
        std::vector<double> oldPriors(m, 1.0);
        // std::vector<double> oldPriors = {1, 1, 1, 1};

        // std::vector<std::vector<double>> oldTransitionMatrix = {
        //     {1.1, 1.2, 1.3},
        //     {2.1, 2.2, 2.3},
        //     {3.1, 3.2, 3.3},
        // };

        std::vector<std::vector<double>> oldTransitionMatrix(
            m, std::vector<double>(m, 2));

        assert(m == oldPriors.size());
        assert(m == oldTransitionMatrix.size());
        assert(m == oldTransitionMatrix.at(0).size());

        //*********************************************************
        // Load In Data
        // *********************************************************

        // Specify CSV Path
        std::string csv_path = "../../../data/synthetic/test.csv";

        // Load Data
        debugPrint(DEBUG,
                   "Loading Drought Indices From '" + csv_path + "' ...");
        std::vector<std::vector<float>> observedAttributes;
        try {
            observedAttributes = readCSV(csv_path);
        } catch (std::exception &e) {
            std::cerr << "ERROR: " << e.what() << '\n';
            return 1;
        }

        // Print Attributes
        if (DEBUG) print2DArray(observedAttributes);

        // ============ Populate Useful Variables Here, Now  ============
        size_t T = observedAttributes.size();
        size_t N = observedAttributes.at(0).size();
        debugPrint(DEBUG, "# Time Steps → T = " + std::to_string(T));
        debugPrint(DEBUG, "# Attribute RVs → N = " + std::to_string(N));

        debugPrint(DEBUG, "✓ Success!\n");

        //*********************************************************
        // Generate RV IDs
        // *********************************************************

        debugPrint(DEBUG, "Creating RV IDs...");

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
        if (!validateRvIds(DEBUG, droughtStateRvIDs, attributeRvIds,
                           observedAttributes)) {
            std::cerr << "ERROR: Generation Of RV IDs Failed...\n";
            return 1;
        }

        debugPrint(DEBUG, "✓ Success!\n");

        //*********************************************************
        // Create Domain Of RVs
        // *********************************************************
        //  - We need a `rcptr` of a vector of ints
        //  - This vector of ints will represent all the possible values the
        //  RV
        //      can take on.
        //  - This is quite simple but still out sourcing to a sperate
        //  function
        //  - Note that this is only for Discrete Factors... (I think)

        debugPrint(DEBUG, "Creating Domains Of The RVs Of The Model...");

        // Hidden Drought State is simply `[1, 2, ..., m]`
        rcptr<std::vector<int>> droughtStateDomain = createDiscreteRvDomain(m);

        // NOTE: This step will change if our attribute RVs are cts, ie.
        // This is Discerete Case. Here we have a vector of
        // `rcptr<std::vector<int>>`s of course indexed by each attribute
        // (Note: This is only for discrete A_t^n)
        std::vector<rcptr<std::vector<int>>> attributeRvDomains;

        // Along columns, thus `maxVals.size()` == N
        std::vector<float> maxVals = findMaxAlongAxis(observedAttributes, 0);

        for (size_t n = 0; n < N; n++) {
            // Convert to int
            int maxVal = static_cast<int>(maxVals.at(n));
            attributeRvDomains.push_back(createDiscreteRvDomain(maxVal));
        }

        printDomainDroughtState(DEBUG, droughtStateDomain);
        printDomainAttributeRVsDiscerete(DEBUG, attributeRvDomains);

        debugPrint(DEBUG, "✓ Success!\n");

        //*********************************************************
        // Define Factors Of Model
        // *********************************************************
        //  - Here we get the factors of the model, ie. the probability
        //  tables
        //      and things

        debugPrint(DEBUG, "Creating Factors Of The Model...");

        // ==================== Priors → p(S_1) ====================
        rcptr<Factor> pS1 = createPriorS1Factor(
            droughtStateDomain, oldPriors, droughtStateRvIDs.at(0), priorNoise);

        if (DEBUG) printFactor(*pS1, "p(S_1)");

        // ================= Transition → p(S_{t+1} | S_t) =================

        std::vector<rcptr<Factor>> transitionFactors = createTransitionFactors(
            droughtStateDomain, oldTransitionMatrix, droughtStateRvIDs);

        if (DEBUG) printFactor(*transitionFactors.at(0), "p(S_2 | S_1)");

        // Change this flag for very comprehensive test
        if (false) {
            if (DEBUG) printFactor(*transitionFactors.at(1), "p(S_3 | S_2)");
            if (DEBUG)
                printFactor(*transitionFactors.at(transitionFactors.size() - 1),
                            "p(S_" + std::to_string(T) + " | S_" +
                                std::to_string(T - 1) + ")");
            print2DArray(oldTransitionMatrix);
        }

        // ================= Emission → p(A_t^n | S_t) =================
        // NOTE: This step will change if our attribute RVs are cts, ie.
        // This is Discerete Case.

        // Create Parameters
        std::vector<std::vector<std::vector<double>>> oldEmissionProbs =
            createEmissionParamsDiscrete(attributeRvDomains, m);

        assert(oldEmissionProbs.size() == N);

        if (DEBUG) {
            for (int n = 0; n < N; n++) {
                print2DArray(oldEmissionProbs.at(n),
                             "Params For A^" + std::to_string(n + 1));
                std::cout << '\n';
            }
        }

        // Create Factors
        std::vector<std::vector<rcptr<Factor>>> emissionFactors =
            createEmissionFactorsDiscrete(droughtStateRvIDs, attributeRvIds,
                                          droughtStateDomain,
                                          attributeRvDomains, oldEmissionProbs);

        if (DEBUG) printFactor(*emissionFactors.at(0).at(1), "p(A^2_1 | S_1)");

        debugPrint(DEBUG, "✓ Success!\n");
        //*********************************************************
        // EM ALGORITHM!!!!
        // *********************************************************

        debugPrint(DEBUG, "Performing EM Algorithm...");

        // ==================== Initialise Loop Params ====================
        size_t numIter = 1;

        // ==================== Timer ====================
        auto startEM = std::chrono::high_resolution_clock::now();
        auto endEM = std::chrono::high_resolution_clock::now();

        while (1) {
            // ============= Obtain p(H|D, Theta) (E-Step) ============
            std::pair<ClusterGraph, std::map<Idx2, rcptr<Factor>>> outp =
                performLBU_LTRIP_discrete(pS1, transitionFactors,
                                          emissionFactors, droughtStateRvIDs,
                                          attributeRvIds, observedAttributes);
            auto &[cg, msgs] = outp;

            // ================ Parameter Update (M-Step) ================

            // TODO: Doing Incredibly Naive Approach With Gross Number Of
            // Loops,
            //  Will Clean Later

            // Priors (π_j)
            std::vector<double> newPriors;
            rcptr<Factor> qS1 =
                queryLBU_CG(cg, msgs, {droughtStateRvIDs.at(0)})->normalize();
            for (int i = 1; i <= m; i++) {
                newPriors.push_back(qS1->potentialAt({0}, {int(i)}));
            }

            if (newPriors.size() != oldPriors.size())
                throw std::runtime_error("Update Rule For Priors Failed...");

            // Transition Probs (a_{i,j})
            std::vector<std::vector<double>> newTransitionMatrix;

            // Using this `i` as values, thus is `int` & begins at 1 instead
            // of
            //  `size_t` that begins at 0
            for (int i = 1; i <= m; i++) {
                std::vector<double> newRow;
                // Denominator
                double runningSumOverTm1_qStEi = 0.0;
                for (size_t t = 0; t < T - 1; t++) {
                    rcptr<Factor> qSt =
                        queryLBU_CG(cg, msgs, {droughtStateRvIDs.at(t)})
                            ->normalize();
                    runningSumOverTm1_qStEi +=
                        qSt->potentialAt({droughtStateRvIDs.at(t)}, {i});
                }
                // Same Story as for `i`
                for (int j = 1; j <= m; j++) {
                    // Numerator
                    double runningSumOverTm1_qStEiJqStp1Ej = 0.0;
                    for (size_t t = 0; t < T - 1; t++) {
                        // Sum over q(S_t = i, S_{t+1} = j)
                        rcptr<Factor> qStStp1 =
                            queryLBU_CG(cg, msgs,
                                        {droughtStateRvIDs.at(t),
                                         droughtStateRvIDs.at(t + 1)})
                                ->normalize();

                        runningSumOverTm1_qStEiJqStp1Ej +=
                            qStStp1->potentialAt({droughtStateRvIDs.at(t),
                                                  droughtStateRvIDs.at(t + 1)},
                                                 {i, j});
                    }
                    newRow.push_back(runningSumOverTm1_qStEiJqStp1Ej /
                                     runningSumOverTm1_qStEi);
                }
                newTransitionMatrix.push_back(newRow);
            }

            // Size check
            if ((newTransitionMatrix.size() != oldTransitionMatrix.size()) ||
                (newTransitionMatrix.at(0).size() !=
                 oldTransitionMatrix.at(0).size()))
                throw std::runtime_error(
                    "Update Rule For Transition Probs Failed...");

            // Emission Probs (b_i^(n)(j))
            std::vector<std::vector<std::vector<double>>> newEmissionProbs;

            for (size_t n = 0; n < N; n++) {
                size_t Cn = attributeRvDomains.at(n)->size();
                std::vector<std::vector<double>> inpMatrix(
                    Cn, std::vector<double>(m, 0.0));

                for (int i = 1; i <= m; i++) {
                    // Denominator
                    double runningSumOverT_qStEi = 0.0;
                    for (size_t t = 0; t < T; t++) {
                        rcptr<Factor> qSt =
                            queryLBU_CG(cg, msgs, {droughtStateRvIDs.at(t)})
                                ->normalize();
                        runningSumOverT_qStEi +=
                            qSt->potentialAt({droughtStateRvIDs.at(t)}, {i});
                    }

                    for (int j = 1; j <= Cn; j++) {
                        // Numerator
                        double runningSumOverT_qStEi_with_AntEj = 0.0;
                        for (size_t t = 0; t < T; t++) {
                            if (observedAttributes.at(t).at(n) == j) {
                                rcptr<Factor> qSt =
                                    queryLBU_CG(cg, msgs,
                                                {droughtStateRvIDs.at(t)})
                                        ->normalize();
                                runningSumOverT_qStEi_with_AntEj +=
                                    qSt->potentialAt({droughtStateRvIDs.at(t)},
                                                     {i});
                            }  // End of if
                        }  // End of t loop

                        inpMatrix[j - 1][i - 1] =
                            runningSumOverT_qStEi_with_AntEj /
                            runningSumOverT_qStEi;
                    }  // End of j loop
                }  // End of i loop
                newEmissionProbs.push_back(inpMatrix);
            }  // End of n loop

            // Size check
            if ((newEmissionProbs.size() != oldEmissionProbs.size()) ||
                (newEmissionProbs.size() != N))
                throw std::runtime_error(
                    "Update Rule For Emission Probs Failed...");

            for (size_t n = 0; n < N; n++) {
                size_t Cn = attributeRvDomains.at(n)->size();

                // std::cout << "n = " << n << '\n';
                // std::cout << "Old B_n = (" << oldEmissionProbs.at(n).size()
                //           << ", " << oldEmissionProbs.at(n).at(0).size()
                //           << ")\n";

                // std::cout << "New B_n = (" << newEmissionProbs.at(n).size()
                //           << ", " << newEmissionProbs.at(n).at(0).size()
                //           << ")\n";

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
            std::cout << "Iteration " << numIter << '\n';
            printVector(oldPriors, "Old Priors");
            printVector(newPriors, "New Priors");

            print2DArray(oldTransitionMatrix, "Old Transition Matrix");
            print2DArray(newTransitionMatrix, "New Transition Matrix");

            // TODO:
            std::cout << "Not printing emission probs...";

            std::cout << "------------------------\n\n";

            // ============= Iterate Loop & Exit Condition =============
            numIter += 1;
            if (numIter > maxIters) {
                // End Timer
                endEM = std::chrono::high_resolution_clock::now();

                debugPrint(DEBUG, "Saving Output");
                saveModelOutput("../../../data/synthetic/output.csv", cg, msgs,
                                m, droughtStateRvIDs);
                break;
            }

            // ============= Update Old Params For Next Loop
            oldPriors = newPriors;
            oldTransitionMatrix = newTransitionMatrix;
            oldEmissionProbs = newEmissionProbs;
            // ============= Create New Model (New Factors) =============
            // Create new node factors

            // Priors
            pS1 = createPriorS1Factor(droughtStateDomain, oldPriors,
                                      droughtStateRvIDs.at(0), priorNoise);
            // Transition

            transitionFactors = createTransitionFactors(
                droughtStateDomain, oldTransitionMatrix, droughtStateRvIDs);

            // Emission
            // NOTE: This is Discerete Case.
            emissionFactors = createEmissionFactorsDiscrete(
                droughtStateRvIDs, attributeRvIds, droughtStateDomain,
                attributeRvDomains, oldEmissionProbs);
        }

        // ==================== End Timer ====================
        std::chrono::duration<double> elapsedEM = endEM - startEM;

        // ==================== Display Info ====================
        std::cout << '\n'
                  << numIter << " Iterations of EM algorithm completed in : "
                  << elapsedEM.count() << " seconds\n\n";

        debugPrint(DEBUG, "✓ Success");

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
    //
}

void debugPrint(bool debug, std::string message) {
    if (debug) std::cout << message << '\n';
}

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

// TODO: CHANGE FLOATS TO INTS SINCE WE STAYING DISCRETE NOW
/**
 * @brief
 *  - Reads in Drought Inidice Attributes.
 *  - Must be in Form (A^1, A^2, A^3, ....)
 * @param
 *  -
 * @return
 *  - 2D Array (Number Attributes, Value At Each Time Step) → (N, T)
 */
std::vector<std::vector<float>> readCSV(std::string_view filePath,
                                        bool hasHeader) {
    std::ifstream fin;
    std::string line;
    std::vector<std::vector<float>> outp;

    fin.open(filePath);
    if (!fin) {
        throw std::runtime_error("Given `filePath = " + std::string(filePath) +
                                 "` could not be opened...");
    }

    // Skip a Line If There Is a Header
    if (hasHeader) std::getline(fin, line);

    while (std::getline(fin, line)) {
        std::vector<std::string> lineElements = splitByCharacter(line, ',');

        std::vector<float> timeStep;
        for (size_t i = 0; i < lineElements.size(); i++) {
            float value = std::stof(lineElements.at(i));
            timeStep.push_back(value);
        }
        outp.push_back(timeStep);
    }

    fin.close();
    return outp;
}

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
                   const std::vector<std::vector<float>> &observedAttributes) {
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
std::vector<std::vector<std::vector<double>>> createEmissionParamsDiscrete(
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
std::vector<std::vector<rcptr<Factor>>> createEmissionFactorsDiscrete(
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
std::pair<ClusterGraph, std::map<Idx2, rcptr<Factor>>>
performLBU_LTRIP_discrete(
    const rcptr<Factor> &pS1,
    const std::vector<rcptr<Factor>> &transitionFactors,
    const std::vector<std::vector<rcptr<Factor>>> &emissionFactors,
    const std::vector<emdw::RVIdType> &droughtStateRvIDs,
    const std::vector<std::vector<emdw::RVIdType>> &attributeRvIds,
    const std::vector<std::vector<float>> &observedAttributes) {
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

    // TODO: Ask about this `->getVars()` method
    // for (size_t t = 0; t < T - 1; t++) {
    //     factorsVector.push_back(uniqptr<Factor>(
    //         transitionFactors.at(t)->copy(transitionFactors.at(t)->getVars())));
    // }

    // This `getVars()` Version is Fine? TODO: ASK ABOUT THIS
    // for (size_t t = 0; t < T; t++) {
    //     for (size_t n = 0; n < N; n++) {
    //         factorsVector.push_back(
    //             uniqptr<Factor>(emissionFactors.at(t).at(n)->copy(
    //                 emissionFactors.at(t).at(n)->getVars())));

    //         std::cout << "Inserted p(A^" << n + 1 << '_' << t + 1 << " |
    //         S_"
    //                   << t + 1 << ")\n";

    //         std::cout << emissionFactors.at(t).at(n)->getVars() <<
    //         std::endl; std::cout << attributeRvIds.at(t).at(n) << ", "
    //                   << droughtStateRvIDs.at(t) << std::endl;
    //     }
    // }

    // ================ Assign All Observed Variables ================
    // The result is a `std::map` where the key is the ID of the observed RV
    //  while the value is the observed data
    std::map<emdw::RVIdType, AnyType> observedData;

    for (size_t t = 0; t < T; t++) {
        for (size_t n = 0; n < N; n++) {
            observedData[attributeRvIds.at(t).at(n)] =
                static_cast<int>(observedAttributes.at(t).at(n));
        }
    }

    // ==================== The Rest Is Just Copied ====================

    // Create Clustergraph
    ClusterGraph cg(ClusterGraph::LTRIP, factorsVector, observedData);
    /* std::cout << cg << std::endl; */

    // export the graph to graphviz .dot format
    // cg.exportToGraphViz("hamming74");

    // Now Calibrate the graph
    std::map<Idx2, rcptr<Factor>> msgs;
    MessageQueue msgQ;

    msgs.clear();
    msgQ.clear();
    unsigned nMsgs = loopyBU_CG(cg, msgs, msgQ);
    std::cout << "Sent " << nMsgs << " messages before convergence\n";

    return std::make_pair(cg, msgs);
}

void saveModelOutput(std::string_view filePath, ClusterGraph &cg,
                     std::map<Idx2, rcptr<Factor>> &msgs, const int m,
                     const std::vector<emdw::RVIdType> &droughtStateRvIDs) {
    size_t T = droughtStateRvIDs.size();

    std::ofstream fout;
    fout.open(filePath);
    if (!fout)
        throw std::invalid_argument("Could Not Open: `" +
                                    std::string(filePath) + '`');

    fout << "St\n";

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

        // Populate CSV with value
        fout << maxVal << '\n';
    }
    fout.close();
}
