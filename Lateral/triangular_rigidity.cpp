// triangular_rigidity_noRotate_crack.cpp
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <random>
#include <algorithm>
#include <omp.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef Vector2d Vec2;
typedef vector<Vec2> Vec2Array;

// -----------------------------------------------------------------------------
// 1) Structures and Global Parameters
// -----------------------------------------------------------------------------

// Bond: tag==0 => horizontal, tag==1 => diagonal
struct LabeledBond {
    int i, j;
    int tag;
};

// Forward declarations for functions that use LabeledBond:
Vec2 computeCenter(const Vec2Array &pos);
bool isInsideEllipse(const Vec2 &p, const Vec2 &center, double a, double b);
vector<LabeledBond> removeCrackBonds_endpoints(const vector<LabeledBond> &bonds,
                                               const Vec2Array &pos,
                                               double crack_a,
                                               double crack_b,
                                               const Vec2 &center);
void applyTensionNoPoissonBC(Vec2Array &pos, int Nx, int Ny, double stretchFactor, struct System &sys);

// Rest lengths.
const double L0   = 1.0;    // diagonal target length
const double L0_h = 1.0;    // horizontal target length (unstretched)

// New extension factors.
// Each horizontal spring now targets 1.25 times its initial rest length.
const double horizontalExtensionTargetFactor = 1.25;
// But the overall lattice width is constrained to be only 1.10 times the original.
const double globalHorizontalStretchFactor = 1.15;

// New bond dilution probabilities.
// p_target_x applies to horizontal bonds (tag 0)
// p_target_y applies to diagonal bonds (tag 1)
const double p_target_x = 0.9;
const double p_target_y = 1.0;

const double c_corr   = 0.0; // correlation parameter

// Crack parameters: Now the crack is vertical (minor axis along x, major along y).
const double crack_a = 2.0;  // semi-axis in x (small)
const double crack_b = 10.0; // semi-axis in y (large)

// Lattice & simulation parameters.
const double alpha = 0.1;
const int Nx = 90;
const int Ny = 90;
const int nRealizations = 5;
const double eps = 0.0;
const double A_unit = sqrt(3.0)/2.0;

// Random generator.
mt19937 rng((unsigned)time(NULL));

// -----------------------------------------------------------------------------
// 2) Build the Lattice (Rectangular Outline) Explicitly
// -----------------------------------------------------------------------------

Vec2Array buildTriangularLatticeNoPeriodic(int Nx, int Ny) {
    Vec2Array pos;
    pos.reserve(Nx * Ny);
    double dx = 1.0;
    double dy = sqrt(3.0) / 2.0;
    for (int j = 0; j < Ny; j++) {
        double xOffset = (j % 2 == 1) ? 0.5 * dx : 0.0;
        for (int i = 0; i < Nx; i++) {
            double x = i * dx + xOffset;
            double y = j * dy;
            pos.push_back(Vec2(x, y));
        }
    }
    return pos;
}

// -----------------------------------------------------------------------------
// 3) Build Bonds Explicitly (No Distance Checks)
// -----------------------------------------------------------------------------

vector<LabeledBond> buildTriangularBondsNoPeriodic(const Vec2Array &pos, int Nx, int Ny) {
    vector<LabeledBond> bonds;
    bonds.reserve(3 * Nx * Ny);
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            int idx = j * Nx + i;
            // Horizontal neighbor.
            if (i + 1 < Nx) {
                int idx_right = j * Nx + (i + 1);
                bonds.push_back({idx, idx_right, 0});
            }
            // Diagonals in the next row.
            if (j + 1 < Ny) {
                if (j % 2 == 0) {
                    if (i - 1 >= 0) {
                        int idx_ul = (j + 1) * Nx + (i - 1);
                        bonds.push_back({idx, idx_ul, 1});
                    }
                    int idx_ur = (j + 1) * Nx + i;
                    bonds.push_back({idx, idx_ur, 1});
                } else {
                    int idx_ul = (j + 1) * Nx + i;
                    bonds.push_back({idx, idx_ul, 1});
                    if (i + 1 < Nx) {
                        int idx_ur = (j + 1) * Nx + (i + 1);
                        bonds.push_back({idx, idx_ur, 1});
                    }
                }
            }
        }
    }
    return bonds;
}

// -----------------------------------------------------------------------------
// 3.5) Crack Implementation: Remove Bonds if EITHER Endpoint is inside Ellipse
// -----------------------------------------------------------------------------

Vec2 computeCenter(const Vec2Array &pos) {
    Vec2 center(0, 0);
    for (auto &p : pos) {
        center += p;
    }
    center /= pos.size();
    return center;
}

bool isInsideEllipse(const Vec2 &p, const Vec2 &center, double a, double b) {
    double val = pow((p[0] - center[0]) / a, 2) + pow((p[1] - center[1]) / b, 2);
    return (val <= 1.0);
}

vector<LabeledBond> removeCrackBonds_endpoints(const vector<LabeledBond> &bonds,
                                               const Vec2Array &pos,
                                               double crack_a,
                                               double crack_b,
                                               const Vec2 &center) {
    vector<LabeledBond> new_bonds;
    for (const auto &bond : bonds) {
        bool inside_i = isInsideEllipse(pos[bond.i], center, crack_a, crack_b);
        bool inside_j = isInsideEllipse(pos[bond.j], center, crack_a, crack_b);
        if (!inside_i && !inside_j) {
            new_bonds.push_back(bond);
        }
    }
    return new_bonds;
}

// -----------------------------------------------------------------------------
// 4) Uncorrelated Dilution (iterative, c=0 => uniform probability)
// -----------------------------------------------------------------------------

vector<LabeledBond> diluteBondsCorrelated(const vector<LabeledBond> &bonds,
                                            double p_target,
                                            double c,
                                            mt19937 &local_rng) {
    vector<LabeledBond> kept;
    vector<LabeledBond> candidates = bonds;
    shuffle(candidates.begin(), candidates.end(), local_rng);
    int total = (int)bonds.size();
    int targetCount = (int)(p_target * total);
    int n_max = 6;
    bool added;
    while ((int)kept.size() < targetCount) {
        added = false;
        for (auto &bond : candidates) {
            bool already = false;
            for (auto &k : kept) {
                if (bond.i == k.i && bond.j == k.j && bond.tag == k.tag) {
                    already = true;
                    break;
                }
            }
            if (already)
                continue;
            int n = 0;
            for (auto &k : kept) {
                if (bond.i == k.i || bond.i == k.j || bond.j == k.i || bond.j == k.j)
                    n++;
            }
            double P = pow(1.0 - c, double(n_max - n));
            double roll = uniform_real_distribution<double>(0.0, 1.0)(local_rng);
            if (roll < P) {
                kept.push_back(bond);
                added = true;
                if ((int)kept.size() >= targetCount)
                    break;
            }
        }
        if (!added)
            break;
    }
    return kept;
}

// -----------------------------------------------------------------------------
// New Function: Dilution with Two Probabilities
// -----------------------------------------------------------------------------

vector<LabeledBond> diluteBondsCorrelated2(const vector<LabeledBond> &bonds,
                                             double p_target_x,
                                             double p_target_y,
                                             double c,
                                             mt19937 &local_rng) {
    vector<LabeledBond> horiz;
    vector<LabeledBond> diag;
    for (const auto &bond : bonds) {
        if (bond.tag == 0)
            horiz.push_back(bond);
        else
            diag.push_back(bond);
    }
    vector<LabeledBond> kept_h = diluteBondsCorrelated(horiz, p_target_x, c, local_rng);
    vector<LabeledBond> kept_d = diluteBondsCorrelated(diag, p_target_y, c, local_rng);
    vector<LabeledBond> kept;
    kept.insert(kept.end(), kept_h.begin(), kept_h.end());
    kept.insert(kept.end(), kept_d.begin(), kept_d.end());
    return kept;
}

// -----------------------------------------------------------------------------
// 5) System Structure for Minimization
// -----------------------------------------------------------------------------

struct System {
    Vec2Array pos;
    vector<int> freeIdx;
    vector<int> fixedIdx;
};

// -----------------------------------------------------------------------------
// 6) New Boundary Conditions: Apply Tension with No Poisson Effect
// -----------------------------------------------------------------------------

// This function fixes all boundaries:
// - Left edge: x fixed at 0.
// - Right edge: x fixed at (Nx-1)*stretchFactor.
// - Bottom edge: y fixed.
// - Top edge: y fixed.
void applyTensionNoPoissonBC(Vec2Array &pos, int Nx, int Ny, double stretchFactor, System &sys) {
    sys.freeIdx.clear();
    sys.fixedIdx.clear();
    int N = Nx * Ny;
    vector<int> dofType(2 * N, 0); // 0 => free, 1 => fixed

    // Fix left edge: nodes with i == 0, set x = 0.
    for (int j = 0; j < Ny; j++) {
        int node = j * Nx;
        pos[node][0] = 0.0;
        dofType[2 * node + 0] = 1;
    }
    // Fix right edge: nodes with i == Nx-1, set x = (Nx-1)*stretchFactor.
    for (int j = 0; j < Ny; j++) {
        int node = j * Nx + (Nx - 1);
        pos[node][0] = (Nx - 1) * stretchFactor;
        dofType[2 * node + 0] = 1;
    }
    // Fix bottom edge: nodes with j == 0, fix y.
    for (int i = 0; i < Nx; i++) {
        int node = i;
        dofType[2 * node + 1] = 1;
    }
    // Fix top edge: nodes with j == Ny-1, fix y.
    for (int i = 0; i < Nx; i++) {
        int node = (Ny - 1) * Nx + i;
        dofType[2 * node + 1] = 1;
    }
    // Build fixed and free DOFs.
    for (int node = 0; node < N; node++) {
        for (int comp = 0; comp < 2; comp++) {
            int dofIdx = 2 * node + comp;
            if (dofType[dofIdx] == 1)
                sys.fixedIdx.push_back(dofIdx);
            else
                sys.freeIdx.push_back(dofIdx);
        }
    }
    sys.pos = pos;
}

// -----------------------------------------------------------------------------
// 7) Energy & Gradient
// -----------------------------------------------------------------------------

// Modified: horizontal bonds now have target = L0_h * horizontalExtensionTargetFactor.
double energyAndGrad(const Vec2Array &pos,
                     const vector<LabeledBond> &bonds,
                     VectorXd &grad,
                     const vector<int> &freeDofs) {
    int N = (int)pos.size();
    vector<Vec2> gradAll(N, Vec2(0, 0));
    double E = 0.0;
    for (auto &b : bonds) {
        Vec2 dr = pos[b.j] - pos[b.i];
        double target = (b.tag == 0) ? L0_h * horizontalExtensionTargetFactor : L0;
        double r = dr.norm();
        double diff = r - target;
        E += 0.5 * diff * diff;
        if (r > 1e-10) {
            Vec2 f = (diff / r) * dr;
            gradAll[b.i] -= f;
            gradAll[b.j] += f;
        }
    }
    grad = VectorXd::Zero(freeDofs.size());
    for (int k = 0; k < (int)freeDofs.size(); k++) {
        int dofIdx = freeDofs[k];
        int node = dofIdx / 2;
        int comp = dofIdx % 2;
        grad[k] = gradAll[node][comp];
    }
    return E;
}

// -----------------------------------------------------------------------------
// 8) Conjugate Gradient Minimizer
// -----------------------------------------------------------------------------

VectorXd minimizeEnergy(System &sys, const vector<LabeledBond> &bonds) {
    int nFree = (int)sys.freeIdx.size();
    VectorXd x(nFree);
    for (int k = 0; k < nFree; k++) {
        int dofIdx = sys.freeIdx[k];
        int node = dofIdx / 2;
        int comp = dofIdx % 2;
        x[k] = sys.pos[node][comp];
    }
    const double tol_grad = 1e-10;
    const int maxIter = 50000;
    const double c1 = 1e-4, c2 = 0.9;
    VectorXd g;
    double f = energyAndGrad(sys.pos, bonds, g, sys.freeIdx);
    VectorXd d = -g;
    int iter = 0;
    while (g.norm() > tol_grad && iter < maxIter) {
        double alpha = 1.0;
        double f0 = f;
        double gTd = g.dot(d);
        int ls_iter = 0;
        while (true) {
            VectorXd x_new = x + alpha * d;
            Vec2Array pos_temp = sys.pos;
            for (int m = 0; m < nFree; m++) {
                int dofIdx = sys.freeIdx[m];
                int node = dofIdx / 2;
                int comp = dofIdx % 2;
                pos_temp[node][comp] = x_new[m];
            }
            VectorXd g_new;
            double f_new = energyAndGrad(pos_temp, bonds, g_new, sys.freeIdx);
            if (f_new <= f0 + c1 * alpha * gTd &&
                fabs(g_new.dot(d)) <= c2 * fabs(gTd)) {
                break;
            }
            alpha *= 0.5;
            ls_iter++;
            if (ls_iter > 50)
                break;
        }
        VectorXd x_new = x + alpha * d;
        for (int m = 0; m < nFree; m++) {
            int dofIdx = sys.freeIdx[m];
            int node = dofIdx / 2;
            int comp = dofIdx % 2;
            sys.pos[node][comp] = x_new[m];
        }
        VectorXd g_new;
        double f_new = energyAndGrad(sys.pos, bonds, g_new, sys.freeIdx);
        double beta = 0.0;
        if (iter > 0) {
            VectorXd g_diff = g_new - g;
            beta = max(0.0, g_new.dot(g_diff) / (g.dot(g) + 1e-12));
        }
        x = x_new;
        f = f_new;
        g = g_new;
        d = -g + beta * d;
        iter++;
    }
    return VectorXd::Zero(nFree);
}

// -----------------------------------------------------------------------------
// 9) Save Network
// -----------------------------------------------------------------------------

void saveNetwork(const string &filename,
                 const Vec2Array &pos,
                 const vector<LabeledBond> &bonds) {
    ofstream ofs(filename);
    ofs << "# Node index, x, y\n";
    for (int i = 0; i < (int)pos.size(); i++) {
        ofs << i << " " << pos[i][0] << " " << pos[i][1] << "\n";
    }
    ofs << "\n# Bonds (node indices, tag):\n";
    for (auto &b : bonds) {
        ofs << b.i << " " << b.j << " " << b.tag << "\n";
    }
    ofs.close();
}

// -----------------------------------------------------------------------------
// 10) Single Run for a Given Simulation (Using p_target_x and p_target_y)
// -----------------------------------------------------------------------------

double simulateOne(const Vec2Array &pos0, int Nx, int Ny, double stretchFactor) {
    // Build full bond adjacency.
    vector<LabeledBond> bonds_full = buildTriangularBondsNoPeriodic(pos0, Nx, Ny);
    // Compute network center.
    Vec2 center = computeCenter(pos0);
    // Remove bonds if either endpoint is inside the ellipse (clean vertical crack).
    bonds_full = removeCrackBonds_endpoints(bonds_full, pos0, crack_a, crack_b, center);
    // Dilute bonds using two different probabilities.
    mt19937 local_rng((unsigned)time(NULL) + omp_get_thread_num());
    vector<LabeledBond> bonds_active = diluteBondsCorrelated2(bonds_full, p_target_x, p_target_y, c_corr, local_rng);
    
    // System setup.
    System sys;
    // Save the unstretched initial configuration.
    {
        ostringstream fname;
        fname << "unstretched_initial_network_px_" << p_target_x 
              << "_py_" << p_target_y << ".dat";
        saveNetwork(fname.str(), pos0, bonds_active);
    }
    
    // Apply boundary conditions (fix all edges) to impose horizontal tension.
    sys.pos = pos0;
    applyTensionNoPoissonBC(sys.pos, Nx, Ny, stretchFactor, sys);
    
    // Save the stretched (but not yet minimized) configuration.
    {
        ostringstream fname;
        fname << "stretched_initial_network_px_" << p_target_x 
              << "_py_" << p_target_y << ".dat";
        saveNetwork(fname.str(), sys.pos, bonds_active);
    }
    
    // Minimize energy.
    minimizeEnergy(sys, bonds_active);
    
    // Compute final energy.
    double E = 0.0;
    for (auto &b : bonds_active) {
        Vec2 dr = sys.pos[b.j] - sys.pos[b.i];
        double target = (b.tag == 0) ? L0_h * horizontalExtensionTargetFactor : L0;
        double r = dr.norm();
        double diff = r - target;
        E += 0.5 * diff * diff;
    }
    double A = (Nx - 1) * (Ny - 1) * A_unit;
    double G = 2.0 * E / A; // Shear modulus measure (can be ignored)
    return G;
}

// -----------------------------------------------------------------------------
// 11) main()
// -----------------------------------------------------------------------------

int main(){
    srand(time(0));
    // Build the triangular lattice (natural, unstretched configuration).
    Vec2Array pos0 = buildTriangularLatticeNoPeriodic(Nx, Ny);
    cout << "Built triangular lattice (no periodic) with " 
         << pos0.size() << " nodes.\n";
    
    // Set the global stretch factor to enforce overall horizontal width.
    double stretchFactor = globalHorizontalStretchFactor;  // 1.10 times the original width.
    
    vector<double> Gvals;
    
    ofstream ofs("shear_modulus_vs_px_" + to_string(p_target_x) +
                 "_py_" + to_string(p_target_y) + ".dat");
    ofs << "# p_target_x  p_target_y  G\n";
    
    double sumG = 0.0;
    for (int r = 0; r < nRealizations; r++) {
        Vec2Array localPos = pos0;
        double G = simulateOne(localPos, Nx, Ny, stretchFactor);
        sumG += G;
    }
    double Gavg = sumG / (double)nRealizations;
    Gvals.push_back(Gavg);
    ofs << p_target_x << " " << p_target_y << " " << Gavg << "\n";
    cout << "p_target_x=" << p_target_x << ", p_target_y=" << p_target_y << " -> G=" << Gavg << "\n";
    
    // Save snapshots for the simulation.
    {
        vector<LabeledBond> bonds_full = buildTriangularBondsNoPeriodic(pos0, Nx, Ny);
        Vec2 center = computeCenter(pos0);
        bonds_full = removeCrackBonds_endpoints(bonds_full, pos0, crack_a, crack_b, center);
        mt19937 snap_rng((unsigned)time(NULL));
        vector<LabeledBond> bonds_active = diluteBondsCorrelated2(bonds_full, p_target_x, p_target_y, c_corr, snap_rng);
        
        System sys;
        sys.pos = pos0;
        // Save the unstretched configuration.
        {
            ostringstream fname;
            fname << "unstretched_initial_network_px_" << p_target_x 
                  << "_py_" << p_target_y << ".dat";
            saveNetwork(fname.str(), sys.pos, bonds_active);
        }
        // Apply boundary conditions for tension (fix all edges).
        applyTensionNoPoissonBC(sys.pos, Nx, Ny, stretchFactor, sys);
        // Save the stretched configuration (before minimization).
        {
            ostringstream fname;
            fname << "stretched_initial_network_px_" << p_target_x 
                  << "_py_" << p_target_y << ".dat";
            saveNetwork(fname.str(), sys.pos, bonds_active);
        }
        // Minimize energy.
        minimizeEnergy(sys, bonds_active);
        // Save the final relaxed configuration.
        {
            ostringstream fname;
            fname << "final_network_px_" << p_target_x 
                  << "_py_" << p_target_y << ".dat";
            saveNetwork(fname.str(), sys.pos, bonds_active);
        }
    }
    
    ofs.close();
    cout << "\nDone.\n";
    return 0;
}

