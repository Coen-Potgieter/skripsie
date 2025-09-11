/*
 * Author     :  (DSP Group, E&E Eng, US)
 * Created on :
 * Copyright  : University of Stellenbosch, all rights retained
 */

// patrec headers
#include "prlite_logging.hpp"  // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "discretetable.hpp"
#include "emdw.hpp"

// standard headers
#include <algorithm>
#include <cctype>    // toupper
#include <iostream>  // cout, endl, flush, cin, cerr
#include <limits>
#include <map>
#include <memory>
#include <string>  // string
#include <unordered_map>
#include <vector>

using namespace std;
using namespace emdw;

bool DEBUG = true;

std::vector<std::vector<float>> readCSV(std::string_view filePath,
                                        bool hasHeader = true);
template <typename T>
void print2DArray(std::vector<std::vector<T>> inp);
void debugPrint(bool debug, std::string message);
std::vector<emdw::RVIdType> createNodeIds(int numNodes, uint &runningIdCount);
rcptr<std::vector<int>> createDiscreteRvDomain(size_t C);

bool validateRvIds(
    const std::vector<emdw::RVIdType> &droughtStateIds,
    const std::vector<std::vector<emdw::RVIdType>> &attributeIds);

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
        int m = 4;

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
            createNodeIds(T, rvIdentity);

        // Will access A_t^n IDs as a 2D matrix
        std::vector<std::vector<emdw::RVIdType>> attributeRvIds;
        for (size_t n = 0; n < N; n++) {
            std::vector<emdw::RVIdType> singleAttributeRvIds =
                createNodeIds(T, rvIdentity);
            attributeRvIds.push_back(singleAttributeRvIds);
        }

        // Check that all values are unique and of type `emdw::RVIdType`

        if (!validateRvIds(droughtStateRvIDs, attributeRvIds)) {
            std::cerr << "ERROR: Generation Of RV IDs Failed...\n";
            return 1;
        }

        debugPrint(DEBUG, "✓ Success!\n");

        //*********************************************************
        // Define Factors
        // *********************************************************

        debugPrint(DEBUG, "Defining Factors Of The Model...");

        // ==================== Domain ====================
        //  - We need a `rcptr` of a vector of ints
        //  - This vector of ints will represent all the possible values the RV
        //  can take on.
        //  - This is quite simple but still out sourcing to a sperate function
        //  - Note that this is only for Discrete Factors... (I think)

        // Hidden Drought State is simply `[1, 2, ..., m]`
        rcptr<std::vector<int>> droughtDomain = createDiscreteRvDomain(m);

        // Here we have a vector of `rcptr<std::vector<int>>`s of course indexed
        // by each attribute (Note: This is only for discrete A_t^n)
        std::vector<rcptr<std::vector<int>>> attributeRvDomains;

        // TODO: COME BACK HERE AND WORK ON THIS!!!!
        for (size_t n = 0; n < N; n++) {
            float maxVal =
                std::max_element() rcptr<std::vector<int>> droughtDomain =
                    createDiscreteRvDomain(m);
        }

        debugPrint(DEBUG, "✓ Success!\n");

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

}  // main

void debugPrint(bool debug, std::string message) {
    if (debug) std::cout << message << '\n';
}

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
 * @param
 *  -
 * @return
 *  -
 */
template <typename T>
void print2DArray(std::vector<std::vector<T>> inp) {
    std::pair<int, int> dims = std::make_pair(inp.size(), inp[0].size());
    std::cout << "Printing 2D Array...\n";

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
        for (size_t i = 1; i < lineElements.size(); i++) {
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
 *  - Returns a vector of emdw RV IDs, incrementing from given starting point
 * @param
 *  -
 *  - numNodes: Number of RV IDs to create
 *  - runningIdCount: Where to begin incrementing from (inclusive)
 * @return
 *  -
 */
std::vector<emdw::RVIdType> createNodeIds(int numNodes, uint &runningIdCount) {
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
 *  - Will simply cycle through each given data structure and see if they are
 * incrementing
 * @param
 *  -
 * @return
 *  - True (if Correct) or False (If Incorrect)
 */
bool validateRvIds(
    const std::vector<emdw::RVIdType> &droughtStateIds,
    const std::vector<std::vector<emdw::RVIdType>> &attributeIds) {
    int lastItem = -1;

    // Cylce Through hidden drought states
    for (const emdw::RVIdType &droughtID : droughtStateIds) {
        if (lastItem != droughtID - 1) return false;
        lastItem += 1;
    }

    debugPrint(DEBUG, "Hidden Drought RVs Are Correctly Initialised");

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
    debugPrint(DEBUG, "Attribute RVs Are Correctly Initialised");
    return true;
}

/**
 * @brief
 *  - This will create a `rcptr` of a vector of ints
 *  - These ints then increment to max C
 *  - Example with C = 3:
 *      output: [1,2,3]
 * @param
 *  -
 * @return
 *  -
 */
rcptr<std::vector<int>> createDiscreteRvDomain(size_t C) {
    rcptr<std::vector<int>> domain(new std::vector<int>());
    for (int i = 1; i <= R; ++i) {
        domain->push_back(i);
    }
    return domain;
}
