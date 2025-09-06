/*
 * Author     :  (DSP Group, E&E Eng, US)
 * Created on :
 * Copyright  : University of Stellenbosch, all rights retained
 */

// patrec headers
#include "prlite_logging.hpp"  // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "emdw.hpp"
#include "discretetable.hpp"

// standard headers
#include <iostream>  // cout, endl, flush, cin, cerr
#include <cctype>  // toupper
#include <string>  // string
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <random>

using namespace std;
using namespace emdw;

//##################################################################
// Some example code. To compile this, go to the emdw/build
// directory and do a:
// cmake ../; make -j7 example
// To run this while in the build directory, do a:
// src/pmr/example
//
// For your own stuff, make a copy of this one to start with. Then
// edit the CMakeLists.txt (also in this directory) by adding your
// new target in the same way as this example.
//##################################################################

int main(int, char *argv[]) {

  // NOTE: this activates logging and unit tests
  initLogging(argv[0]);
  prlite::TestCase::runAllTests();

  try {

    //*********************************************************
    // Some random generator seeding. Just keep this as is
    //*********************************************************

    unsigned seedVal = emdw::randomEngine.getSeedVal();
    cout <<  seedVal << endl;
    emdw::randomEngine.setSeedVal(seedVal);

    //*********************************************************
    // Predefine some types and constants
    //*********************************************************

    typedef int T;                  // The type of the values that the RVs can take on
    typedef DiscreteTable<T> DT;    // DT now is a short-hand for DiscreteTable<int>
    double defProb = 0.0;           // Any unspecified probs will default to this.
    rcptr< vector<T> > binDom (     // Lists the values that a particular RV can take on
        new vector<T>{0,1});

    //*********************************************************
    // Define the RVs
    //*********************************************************

    // The enum statement here predefines two RV ids: the id of X is 0
    // and the id of Y is 1. This is easy enough in very simple
    // problems, for more complex situations involving many RVs this
    // becomes cumbersome and we will need a datastructure such as a
    // map to save the RV ids in. Consult the userguide for more on
    // this.

    enum{X, Y};

    //*********************************************************
    // Set up a discrete factor (in several ways) over two binary
    // RVs specifying that they must have odd parity (i.e. their
    // values will always differ).
    //***************************************************


    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // The most direct declaration. We show this as an example of
    // construction with a basic set of parameters. See the class
    // specific constructor from line 109 in
    // src/emdw-factors/discretetable.hpp for more detail on the
    // exact types of each variable.
    //
    // IMPORTANT: However, you will instead use a dynamic
    // declaration (lower down), because that will allow you to
    // access via its abstract category namely a Factor.
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    DT objXY(
      {X,Y},           // The ids of the two variables (ascending
                       // order). If not presorted, the order of the
                       // variables will get re-arranged in the class.
      {binDom,binDom}, // The domains over which they can vary
      defProb,         // The default probability for unspecified allocations
      {                // The explicitly specified probabilities:
        {{0,1}, 0.5},  // Note: the {0,1} is the allocation to {X,Y},
        {{1,0}, 0.5},  // and the 0.5 is its probabilty.
      } );

    std::cout << __FILE__ << __LINE__ << ": " << objXY << std::endl; // displays the factor

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Same thing, but now via a dynamic object creation. First
    // thing to see is that for dynamic object creation, which
    // allows us to refer to the abstract type of an object
    // (i.e. Factor instead of DiscreteTable), we have to work via
    // pointers. A pointer is a data type that contains an address of
    // where in memory the actual object resides. It "points" to the
    // actual object but, like a finger pointing to an elephant, it is
    // not the object itself.
    //
    // Then note the hierarchy of pointers at play here. The "new"
    // operator creates a raw c-pointer. But for several reasons we
    // don't want to work with that, but instead use smart pointers
    // (which also implicitly gives us garbage collection). So we first
    // grab the raw pointer via a (temporary) uniqptr - it still points
    // to a concrete DiscreteTable. This then gets transferred to a
    // rcptr pointing to the abstract class (i.e. Factor).
    //
    // This all might feel somewhat daunting. ADVICE: Just copy the
    // shape of it, you'll soon get used to it.
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    rcptr<Factor> ptrXY = // final abstract smart pointer (defined in emdw/src/emdw-base/emdw.hpp)
      uniqptr<DT>(        // temporary DT smart pointer (defined in emdw/src/emdw-base/emdw.hpp)
        new DT(           // raw c-pointer pointing to the newly created DT
          {X,Y},
          {binDom,binDom},
          defProb,
          {
            {{0,1}, 0.5},
            {{1,0}, 0.5},
          } ));

    // Note the '*' before ptrXY. That is called derefencing, i.e. use
    // the object the pointer is pointing to, not the pointer itself.
    // Btw, the __FILE__ << __LINE__ business points the filename and
    // line number in it. Useful to know of, also for debugging.
    std::cout << __FILE__ << __LINE__ << ": " << *ptrXY << std::endl; // displays the factor

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Same thing, but now making the structure for the specified
    // (sparse) probabilities explicit. In more complex
    // applications that can not simply be explicitly stated as above,
    // we usually go via this way.
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // The map contains the explicitly specified sparse
    // probabilities. A c++ map is more or less the same thing as a
    // python dictionary. FProb is defined in
    // src/emdw-factors/discretetable.hpp.
    // Mostly you can think of it (and use it) as a double.
    map<vector<T>, FProb> sparseProbs;
    sparseProbs[{0,1}] = 0.5; // the curly braces sticks the assignment into a std::vector<T>
    sparseProbs[vector<T>({1,0})] = 0.5; // another way to do this

    ptrXY = // re-using the smart pointer, the previous object will get garbage collected.
      uniqptr<DT>(new DT({X,Y}, {binDom,binDom}, defProb, sparseProbs ));

    std::cout << __FILE__ << __LINE__ << ": " << *ptrXY << std::endl; // displays the factor

    //*********************************************************
    // Let's try some Factor operations.
    // There is more on FactorOperator in
    // emdw/src/emdw-factors/factoroperator.hpp
    //***************************************************

    //+++++++++++++++++++++++++++++++++
    // Marginalize to find p(Y)
    //+++++++++++++++++++++++++++++++++

    // The marginalize operator below specifies a parameter {Y} as the
    // variable(s) we want to RETAIN. If we wanted to retain both X
    // and Y (somewhat nonsensical in this case), it would have been
    // {X,Y}. The other thing to notice is the "->" operator. It
    // basically takes the pointer to its left - ptrXY in this case,
    // derefences it, and then applies the function/operator to its
    // right to this object -- marginalize in this case.

    rcptr<Factor> ptrY  = ptrXY->marginalize({Y}); // Note, ptrY points to Factor, not DT
    std::cout << __FILE__ << __LINE__ << ": " << *ptrY << std::endl; // displays the factor


    //++++++++++++++++++++++++++++++++++++
    // Division: P(X|Y) = P(X,Y)/p(Y)
    //++++++++++++++++++++++++++++++++++++

    rcptr<Factor> ptrXgY = ptrXY->cancel(ptrY);
    std::cout << __FILE__ << __LINE__ << ": " << *ptrXgY << std::endl; // displays the factor

    //+++++++++++++++++++++++++++++++++++++++
    // Multiplication P(X,Y) = P(X|Y)P(Y)
    //+++++++++++++++++++++++++++++++++++++++

    std::cout << __FILE__ << __LINE__ << ": " << *ptrXgY->absorb(ptrY) << std::endl;


    //++++++++++++++++++++++++++++++++++++++++++
    // You can chain several operations
    //++++++++++++++++++++++++++++++++++++++++++

    std::cout << __FILE__ << __LINE__ << ": " << *ptrXY->cancel(ptrY)->absorb(ptrY)->normalize() << std::endl;

    //+++++++++++++++++++++++++++++++++++++
    // Observing a variable: P(Y|X=0)
    //+++++++++++++++++++++++++++++++++++++

    std::cout << __FILE__ << __LINE__ << ": " << *ptrXY->observeAndReduce({X},{0})->normalize() << std::endl;

    return 0; // tell the world that all is fine
  } // try

  catch (char msg[]) {
    cerr << msg << endl;
  } // catch

  // catch (char const* msg) {
  //   cerr << msg << endl;
  // } // catch

  catch (const string& msg) {
    cerr << msg << endl;
    throw;
  } // catch

  catch (const exception& e) {
    cerr << "Unhandled exception: " << e.what() << endl;
    throw e;
  } // catch

  catch(...) {
    cerr << "An unknown exception / error occurred\n";
    throw;
  } // catch

} // main
