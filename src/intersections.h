#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/normal.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

#if MESH_BOX
__host__ __device__ float boxIntersectionTest(const Ray &r, glm::vec3 &intersectionPoint,
	glm::vec3 &normal, bool &outside, glm::vec3 boxMin, glm::vec3 boxMax) {
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	glm::vec3 invdir = 1.0f / r.direction;
	int sign[3] = { invdir.x < 0, invdir.y < 0, invdir.z < 0 };
	glm::vec3 bounds[2] = { boxMin, boxMax };

	tmin = (bounds[sign[0]].x - r.origin.x) * invdir.x;
	tmax = (bounds[1 - sign[0]].x - r.origin.x) * invdir.x;
	tymin = (bounds[sign[1]].y - r.origin.y) * invdir.y;
	tymax = (bounds[1 - sign[1]].y - r.origin.y) * invdir.y;

	if ((tmin > tymax) || (tymin > tmax))
		return -1;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bounds[sign[2]].z - r.origin.z) * invdir.z;
	tzmax = (bounds[1 - sign[2]].z - r.origin.z) * invdir.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return -1;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	intersectionPoint = r.origin + r.direction * tmin;

	return glm::length(r.origin - intersectionPoint);
}
#endif

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(Triangle triangle, Ray r,
		glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {

	glm::vec3 bary;
	bool success = glm::intersectRayTriangle(r.origin, r.direction, triangle.v0, triangle.v1, triangle.v2, bary);
	if (!success) {
		return -1;
	}

	float t = bary.z;
	bary.z = 1.0f - bary.x - bary.y;

	intersectionPoint = triangle.v0 * bary.x + triangle.v1 * bary.y + triangle.v2 * bary.z;
	normal = glm::triangleNormal(triangle.v0, triangle.v1, triangle.v2);

	return t;
}

/**
* Test intersection between a ray and a triangle mesh.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float meshIntersectionTest(Ray r, Triangle* triangles, int tri_size,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside
#if MESH_BOX
	, glm::vec3 boxMin, glm::vec3 boxMax
#endif
) {
	// TODO: calculate intersections using glm::intersectRayTriangle and hierarchical data structure
	float t = -1;
#if MESH_BOX
	t = boxIntersectionTest(r, intersectionPoint, normal, outside, boxMin, boxMax);
	if (t < 0) {
		return -1;
	}
	t = -1;
#endif

	glm::vec3 tempIntersection;
	glm::vec3 tempNormal;
	float newt;
	for (int i = 0; i < tri_size; ++i) {
		newt = triangleIntersectionTest(triangles[i], r, tempIntersection, tempNormal, outside);
		//glm::vec3 bary;
		//bool success = glm::intersectRayTriangle(r.origin, r.direction, triangles[i].v0, triangles[i].v1, triangles[i].v2, bary);
		if (newt >= 0 && (t < 0 || newt < t)) {
			t = newt;
			intersectionPoint = tempIntersection;
			normal = tempNormal;
		}
	}
	outside = true;
	return t;
}
