//  hierarchy.h
//  Hierarchy for ray cast acceleration

#pragma once

#include "glm/glm.hpp"
#include <vector>

#define MAX_TREE_DEPTH 16

struct MarkerParticle;

struct Bounds {
	glm::vec3 min = glm::vec3(999999.f);
	glm::vec3 max = glm::vec3(-999999.f);
	Bounds(glm::vec3 newMin, glm::vec3 newMax) {
		min = newMin;
		max = newMax;
	}
};

struct TreeNode {
	Bounds bounds;
	std::vector<TreeNode*> children;
	std::vector<MarkerParticle> particles;
};

struct LinearNode {
	Bounds bounds;
	int childOffset[3];
	int particlesOffset;
	int nParticles;
};

float boundsIntersectionTest(Bounds b, Ray r);
TreeNode* buildTree(std::vector<MarkerParticle> particles, int currentDepth, int maxDepth);
int treeSize(TreeNode *node);
int flattenTree(TreeNode *treeNode, std::vector<MarkerParticle> &particles, std::vector<LinearNode> &tree, int *offset);
void deleteTree(TreeNode* node);
