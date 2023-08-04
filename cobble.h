#include <stdio.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------- */


#ifndef DATA_TYPE
#define DATA_TYPE float // Data type to be used for data storage.
#endif // End of the "#ifndef DATA_TYPE" directive.

/* -------------------------------------------------------------------------- */

// Currently the idea is that lazy simply tracks all the tensor operations and that is it.
// There should be some connecting functions like unaryop(x), binaryop(x, y), etc.
// Leave the implementations/details up to the user. 
// This should be a simple skeleton that can easily be expanded.
// Can also add in a traversal/graphing option for a lazy utility.
// This should also deal with backprop if necessary, though the user will have to provide those functions.
#ifdef LAZY
// Need to figure out how I want to give the user the ability to include other
// files to optimize the customizability of my library.


// Can these not be used inline when referencing to the same datatype inside?
typedef struct Tensor Tensor;
typedef union FORWARD FORWARD;
typedef union BACKWARD BACKWARD;

//Should I create a global counter to track how many layers there are for a size of the linearized list?
////Separate trackers for each type of operation?

/* A union containing the possible types of forward tensor operations. */
union FORWARD {
    Tensor *(*forward)(void); // need different names
    Tensor *(*forward)(Tensor *);
    Tensor *(*forward)(Tensor *, Tensor *);
    int r; // a case where there is no forward function to track?
};

/* A union containing the possible types of backward tensor operations. */
union BACKWARD {
    Tensor *(*backward)(void);
    Tensor *(*backward)(Tensor *);
    Tensor *(*backward)(Tensor *, Tensor *);
};

/* The main struct containing the data's information. */
struct Tensor {
    DATA_TYPE *buffer;
    Tensor *parent_one;
    Tensor *parent_two;
    FORWARD *forward_function;
    BACKWARD *backward_function;
    // Tensor* (*forward); //how to get the naming convention right and the pointer correct
    // Tensor* (*backward); //unions? https://stackoverflow.com/questions/16770690/function-pointer-to-different-functions-with-different-arguments-in-c
    int hi;
};

//unary, binary, reduce, ternary, load

///////////////////////////////////////////////////////////////////////
/*                       Function Declaration                        */
///////////////////////////////////////////////////////////////////////

/* ---- Loading Operation ---- */
Tensor *init_Tensor(void);

/* ---- Unary Operation ---- */

/* ---- Binary Operation ---- */
Tensor *add(Tensor *x, Tensor *y); // To be implemented by the user.

/* ---- Ternary Operation ---- */

/* ---- Reduction Operation ---- */

///////////////////////////////////////////////////////////////////////
/*                        Function Definition                        */
///////////////////////////////////////////////////////////////////////
Tensor *add(Tensor *x, Tensor *y) {
    return x;
}

// Check for a memory allocation failure.
Tensor *init_Tensor(void) {
    Tensor *test = (Tensor *)malloc(sizeof(Tensor));
    test->hi = 2;
    return test;
};

#endif // End of the "#ifdef LAZY" directive.