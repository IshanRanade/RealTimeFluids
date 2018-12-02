//  hierarchy.h
//  Hierarchy for ray cast acceleration

#pragma once

#include "glm/glm.hpp"
#include <vector>

#define QUAD_TREE 0
#define MAX_TREE_DEPTH 4

struct MarkerParticle;

struct Bounds {
	glm::vec3 min;
	glm::vec3 max;
    Bounds() {
        min = glm::vec3(999999.f);
        max = glm::vec3(-999999.f);
    }
	Bounds(glm::vec3 newMin, glm::vec3 newMax) {
		min = newMin;
		max = newMax;
	}
};

struct TreeNode {
	Bounds bounds;
	std::vector<TreeNode*> children;
    // integer vector of indices into marker particles
	std::vector<int> particles;
};

struct LinearNode {
	Bounds bounds;
	int childOffset[3];
	int particlesOffset;
	int particleCount;
};

TreeNode* buildTree(std::vector<int>& oldParticles, MarkerParticle* markerParticles, int currentDepth, glm::vec3 boundMin, glm::vec3 boundMax);
int treeSize(TreeNode* node);
int flattenTree(TreeNode* treeNode, std::vector<int>& particles, std::vector<LinearNode>& tree, int* offset);
void deleteTree(TreeNode* node);
