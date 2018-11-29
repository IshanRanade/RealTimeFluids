//  hierarchy.cpp

#include "hierarchy.h"
#include "fluid.h"

// Intersection for node bounds
float boundsIntersectionTest(Bounds b, glm::vec3 rayPos, glm::vec3 rayDir) {
    float tmin = -999999.f;
    float tmax = 999999.f;

    for (int axis = 0; axis < 3; ++axis) {
        float axisDir = rayDir[axis];
        if (axisDir != 0) {
            float t1 = (b.min[axis] - rayPos[axis]) / axisDir;
            float t2 = (b.max[axis] - rayPos[axis]) / axisDir;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin)
                tmin = ta;
            if (tb < tmax)
                tmax = tb;
        }
    }

    if (tmax >= tmin && tmax > 0) {
        return tmin;
    }
    return -1.f;
}

// Recursively construct tree
TreeNode* buildTree(std::vector<MarkerParticle> particles, int currentDepth, glm::vec3 boundMin, glm::vec3 boundMax) {
    TreeNode* node = new TreeNode();
    node->left = nullptr;
    node->right = nullptr;

    Bounds bounds(boundMin, boundMax);
    float zMin = 999999.f;
    float zMax = -999999.f;

    // remove particles that are not in bounds
    for (int i = 0; i < particles.size(); i++) {
        glm::vec3 particleMin = particles[i].worldLocation - glm::vec3(PARTICLE_RADIUS);
        glm::vec3 particleMax = particles[i].worldLocation + glm::vec3(PARTICLE_RADIUS);

        if(particleMin.x > bounds.max.x || particleMax.x < bounds.min.x)
            particles.remove(particles[i--]);
        else if(particleMin.y > bounds.max.y || particleMax.y < bounds.min.y)
            particles.remove(particles[i--]);
        else if(particleMin.z < zMin)
            zMin = particleMin.z;
        else if(particleMax.z > zMax)
            zMax = particleMax.z;
    }

    if (particles.size() == 0) {
        bounds.max = glm::vec3(0);
        bounds.min = glm::vec3(0);
    } else {
        bounds.min.z = zMin;
        bounds.max.z = zMax;
    }
    node->bounds = bounds;
    node->particles = particles;

    if (particles.size() <= 4 || currentDepth > MAX_TREE_DEPTH)
        return node;

    // build child nodes
    glm::vec3 center = bounds.min + ((bounds.max - bounds.min) / 2.0f);
    node->children.push_back(buildTree(particles, currentDepth + 1, bounds.min, glm::vec3(center.x, center.y, bounds.max.z))); // LL
    node->children.push_back(buildTree(particles, currentDepth + 1, glm::vec3(center.x, center.y, bounds.min.z), bounds.max)); // UR
    node->children.push_back(buildTree(particles, currentDepth + 1, glm::vec3(min.x, center.y, bounds.min.z), glm::vec3(center.x, bounds.max.y, bounds.max.z))); // UL
    node->children.push_back(buildTree(particles, currentDepth + 1, glm::vec3(center.x, min.y, bounds.min.z), glm::vec3(bounds.max.x, center.y, bounds.max.z))); // LR
    return node;
}

// Compute size of tree hierarchy
int treeSize(TreeNode* node) {
    if (node->children.empty())
        return 1;

    int count = 0;
    for(int i = 0; i < 4; ++i) {
        count += treeSize(node->children[i]);
    }
    return count + 1;
}

// Flatten hierarchy into array so that it can be passed to GPU
int flattenTree(TreeNode* treeNode, std::vector<MarkerParticle>& particles, std::vector<LinearNode>& tree, int* offset) {
    LinearNode* linearNode = &tree[*offset];
    linearNode->bounds = treeNode->bounds;

    if (treeNode->children.empty()) {
        //leaf
        linearNode->particlesOffset = particles.size();
        linearNode->nParticles = treeNode->particles.size();

        for (MarkerParticle p : treeNode->particles) {
            particles.push_back(p);
        }
    } else {
        // parent
        linearNode->nParticles = 0;

        flattenTree(treeNode->children[0], particles, tree, offset);
        for(int i = 0; i < 3; ++i) {
            linearNode->childOffset[i] = flattenTree(treeNode->children[i + 1], particles, tree, offset);
        }
    }

    return ++(*offset);
}

// Compute size of tree hierarchy
void deleteTree(TreeNode* node) {
    if (node->children.empty())
        delete node;

    for(int i = 0; i < 4; ++i) {
        deleteTree(node->children[i]);
    }
    delete node;
}
