//  hierarchy.cpp

#include "hierarchy.h"
#include "fluid.h"

// Recursively construct tree
TreeNode* buildTree(std::vector<int>& oldParticles, MarkerParticle* markerParticles, int currentDepth, const glm::vec3 boundMin, const glm::vec3 boundMax) {
    TreeNode* node = new TreeNode();

    Bounds bounds(boundMin, boundMax);
    float yMin = 999999.f;
    float yMax = -999999.f;

    // Remove particles that are not in bounds
    std::vector<int>& particles = node->particles;
    if (currentDepth == 1) {
        particles = oldParticles;
    } else {
        for (int i = 0; i < oldParticles.size(); i++) {
            //const glm::vec3 particleMin = markerParticles[oldParticles[i]].worldPosition - glm::vec3(PARTICLE_RADIUS);
            //const glm::vec3 particleMax = markerParticles[oldParticles[i]].worldPosition + glm::vec3(PARTICLE_RADIUS);
            const glm::vec3 particlePos = markerParticles[oldParticles[i]].worldPosition;

            if (particlePos.x <= bounds.max.x && particlePos.x >= bounds.min.x &&
                particlePos.z <= bounds.max.z && particlePos.z >= bounds.min.z) {

                particles.push_back(oldParticles[i]);

                if (particlePos.y < yMin)
                    yMin = particlePos.y;
                else if (particlePos.y > yMax)
                    yMax = particlePos.y;
            }
        }
    }

    if (particles.empty()) {
        bounds.min = glm::vec3(0);
        bounds.max = glm::vec3(0);
    } else {
        bounds.min.y = yMin;
        bounds.max.y = yMax;
        bounds.min -= glm::vec3(PARTICLE_RADIUS * 4.0f);
        bounds.max += glm::vec3(PARTICLE_RADIUS * 4.0f);
    }
    node->bounds = bounds;

    if (particles.size() <= 4 || currentDepth > MAX_TREE_DEPTH)
        return node;

    // Build child nodes
    const glm::vec3 center = bounds.min + ((bounds.max - bounds.min) / 2.0f);
	//node->children.push_back(buildTree(particles, markerParticles, currentDepth + 1, glm::vec3(0), glm::vec3(0)));
    node->children.push_back(buildTree(particles, markerParticles, currentDepth + 1, bounds.min, glm::vec3(center.x, bounds.max.y, center.z))); // LL
    node->children.push_back(buildTree(particles, markerParticles, currentDepth + 1, glm::vec3(center.x, bounds.min.y, center.z), bounds.max)); // UR
    node->children.push_back(buildTree(particles, markerParticles, currentDepth + 1, glm::vec3(bounds.min.x, bounds.min.y, center.z), glm::vec3(center.x, bounds.max.y, bounds.max.z))); // UL
    node->children.push_back(buildTree(particles, markerParticles, currentDepth + 1, glm::vec3(center.x, bounds.min.y, bounds.min.z), glm::vec3(bounds.max.x, bounds.max.y, center.z))); // LR
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
int flattenTree(TreeNode* treeNode, std::vector<int>& particles, std::vector<LinearNode>& tree, int* offset) {
    LinearNode* linearNode = &tree[*offset];
    linearNode->bounds = treeNode->bounds;

    if (treeNode->children.empty()) {
        //leaf
        linearNode->particlesOffset = particles.size();
        linearNode->particleCount = treeNode->particles.size();

        for (int p : treeNode->particles) {
            particles.push_back(p);
        }
    } else {
        // parent
        linearNode->particleCount = -1;

        ++(*offset);
        flattenTree(treeNode->children[0], particles, tree, offset);
        for(int i = 0; i < 3; ++i) {
            linearNode->childOffset[i] = ++(*offset);
            flattenTree(treeNode->children[i + 1], particles, tree, offset);
        }
    }

    return *offset;
}

// Free tree memory
void deleteTree(TreeNode* node) {
    if (!node->children.empty()) {
        for (int i = 0; i < 4; ++i) {
            deleteTree(node->children[i]);
        }
    }
    delete node;
}
