/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "amg.h"


/*--------------------------------------------------------------------------
 * Main driver for AMG
 *--------------------------------------------------------------------------*/

int   main(argc, argv)
int   argc;
char *argv[];
{
   char    *run_name;

   char     file_name[255];
   FILE    *fp;

   Problem *problem;
   Solver  *solver;

   Vector      *u;
   Vector      *f;
   double       stop_tolerance;
   Data        *amgs01_data;
   Data        *wjacobi_data;
   Data        *pcg_data;


   /*-------------------------------------------------------
    * Check that the number of command args is correct
    *-------------------------------------------------------*/

   if (argc < 2)
   {
      fprintf(stderr, "Usage:  amg <run name>\n");
      exit(1);
   }

   /*-------------------------------------------------------
    * Set up globals
    *-------------------------------------------------------*/

   run_name = argv[1];
   NewGlobals(run_name);

   /*-------------------------------------------------------
    * Set up the problem
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.problem.strp", GlobalsInFileName);
   problem = NewProblem(file_name);

   /*-------------------------------------------------------
    * Set up the solver
    *-------------------------------------------------------*/

   sprintf(file_name, "%s.solver.strp", GlobalsInFileName);
   solver = NewSolver(file_name);

   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/
#if 1
   sprintf(file_name, "%s.ysmp", GlobalsOutFileName);
   WriteYSMP(file_name, ProblemA(problem));

   sprintf(file_name, "%s.initu", GlobalsOutFileName);
   WriteVec(file_name, ProblemU(problem));

   sprintf(file_name, "%s.rhs", GlobalsOutFileName);
   WriteVec(file_name, ProblemF(problem));
#endif

   /*-------------------------------------------------------
    * Write initial logging info
    *-------------------------------------------------------*/

   fp = fopen(GlobalsLogFileName, "w");

   fclose(fp);

   /*-------------------------------------------------------
    * Call the solver
    *-------------------------------------------------------*/

   u = ProblemU(problem);
   f = ProblemF(problem);

   stop_tolerance = SolverStopTolerance(solver);
   amgs01_data    = SolverAMGS01Data(solver);
   wjacobi_data   = SolverWJacobiData(solver);
   pcg_data       = SolverPCGData(solver);

   /* call AMGS01 */
   if (SolverType(solver) == 0)
   {
      AMGS01Setup(problem, amgs01_data);

      AMGS01(u, f, stop_tolerance, amgs01_data);
   }

   /* call AMGCG */
   else if (SolverType(solver) == 1)
   {
      AMGS01Setup(problem, amgs01_data);
      PCGSetup(problem, AMGS01, amgs01_data, pcg_data);

      PCG(u, f, stop_tolerance, pcg_data);
   }

   /* call JCG */
   else if (SolverType(solver) == 2)
   {
      WJacobiSetup(problem, wjacobi_data);
      PCGSetup(problem, WJacobi, wjacobi_data, pcg_data);

      PCG(u, f, stop_tolerance, pcg_data);
   }

   /*-------------------------------------------------------
    * Debugging prints
    *-------------------------------------------------------*/
#if 1
   sprintf(file_name, "%s.lastu", GlobalsOutFileName);
   WriteVec(file_name, ProblemU(problem));

   Matvec(-1.0, ProblemA(problem), ProblemU(problem), 1.0, ProblemF(problem));
   sprintf(file_name, "%s.res", GlobalsOutFileName);
   WriteVec(file_name, ProblemF(problem));
#endif

   return 0;
}

