using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AMP
{
    public static class ConfigurableJointExtensions
    {
        /// <summary>
        /// Sets a joint's targetRotation to match a given local rotation.
        /// The joint transform's local rotation must be cached on Start and passed into this method.
        /// </summary>
        public static void SetTargetRotationLocal(this ConfigurableJoint joint, Quaternion targetLocalRotation, Quaternion startLocalRotation)
        {
            if (joint.configuredInWorldSpace)
            {
                Debug.LogError("SetTargetRotationLocal should not be used with joints that are configured in world space. For world space joints, use SetTargetRotation.", joint);
            }
            SetTargetRotationInternal(joint, targetLocalRotation, startLocalRotation, Space.Self);
        }

        /// <summary>
        /// Sets a joint's targetRotation to match a given world rotation.
        /// The joint transform's world rotation must be cached on Start and passed into this method.
        /// </summary>
        public static void SetTargetRotation(this ConfigurableJoint joint, Quaternion targetWorldRotation, Quaternion startWorldRotation)
        {
            if (!joint.configuredInWorldSpace)
            {
                Debug.LogError("SetTargetRotation must be used with joints that are configured in world space. For local space joints, use SetTargetRotationLocal.", joint);
            }
            SetTargetRotationInternal(joint, targetWorldRotation, startWorldRotation, Space.World);
        }

        static void SetTargetRotationInternal(ConfigurableJoint joint, Quaternion targetRotation, Quaternion startRotation, Space space)
        {
            // Calculate the rotation expressed by the joint's axis and secondary axis
            var right = joint.axis;
            var forward = Vector3.Cross(joint.axis, joint.secondaryAxis).normalized;
            var up = Vector3.Cross(forward, right).normalized;
            Quaternion worldToJointSpace = Quaternion.LookRotation(forward, up);

            // Transform into world space
            Quaternion resultRotation = Quaternion.Inverse(worldToJointSpace);

            // Counter-rotate and apply the new local rotation.
            // Joint space is the inverse of world space, so we need to invert our value
            if (space == Space.World)
            {
                resultRotation *= startRotation * Quaternion.Inverse(targetRotation);
            }
            else
            {
                resultRotation *= Quaternion.Inverse(targetRotation) * startRotation;
            }

            // Transform back into joint space
            resultRotation *= worldToJointSpace;

            // Set target rotation to our newly calculated rotation
            joint.targetRotation = resultRotation;
        }
    }
    public class Utils
    {
        public static Quaternion QuatToExp(Vector3 v, float maxLen, float eps = 1e-8f)
        {
            float len = Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if (len > maxLen)
            {
                v *= maxLen / len;
            }

            // ExpMapToQuaternion
            float theta = v.magnitude;
            float outTheta = 0;
            Vector3 outAxis = new Vector3(0, 0, 1);
            if (theta > 1e-6)
            {
                outAxis = v / theta;
                outTheta = NormlaizeAngle(theta);
            }

            float c = Mathf.Cos(outTheta / 2);
            float s = Mathf.Sin(outTheta / 2);
            Quaternion quat = new Quaternion(
                    s * outAxis.x,
                    s * outAxis.y,
                    s * outAxis.z,
                    c);

            return quat;
        }
        public static Vector3 ExpToQuat(Quaternion q)
        {
            float mag = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (Mathf.Abs(mag - 1.0f) > 1e-6f)
            {
                q = q.normalized;
            }

            // 회전각 theta = 2 * acos(w)
            float angle = 2.0f * Mathf.Acos(q.w);

            // sin(theta/2) 계산
            float s = Mathf.Sqrt(1.0f - q.w * q.w);

            // 수치적 안정성을 위한 임계값
            const float epsilon = 1e-6f;
            if (s < epsilon)
            {
                // 회전각이 거의 0인 경우 (또는 매우 작은 각도), 벡터 부분 그대로 반환
                return new Vector3(q.x, q.y, q.z);
            }
            else
            {
                // 회전축 계산: (x, y, z) / sin(theta/2)
                Vector3 axis = new Vector3(q.x / s, q.y / s, q.z / s);
                // exp-map은 회전축에 회전각을 곱한 값
                return axis * angle;
            }
        }

        public static Quaternion quat_exp(Vector3 v, float eps = 1e-8f)
        {
            float halfangle = Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

            if (halfangle < eps)
            {
                return Quaternion.Normalize(new Quaternion(v.x, v.y, v.z, 1.0f));
            }
            else
            {
                float c = Mathf.Cos(halfangle);
                float s = Mathf.Sin(halfangle) / halfangle;
                return new Quaternion(s * v.x, s * v.y, s * v.z, c);
            }
        }

        public static float FMod(float a, float b)
        {
            return a - b * Mathf.Floor(a / b);
        }

        public static float NormlaizeAngle(float theta)
        {
            float normTheta = FMod(theta, 2 * Mathf.PI);
            if (normTheta > Mathf.PI)
            {
                normTheta = -2 * Mathf.PI + normTheta;
            }
            else if (normTheta < -Mathf.PI)
            {
                normTheta = 2 * Mathf.PI + normTheta;
            }
            return normTheta;
        }

        public static bool VectorValidate(Vector3 value)
        {
            return !(float.IsNaN(value.x) || float.IsNaN(value.y) || float.IsNaN(value.z) ||
                     float.IsInfinity(value.x) || float.IsInfinity(value.y) || float.IsInfinity(value.z));
        }


        public static Vector3 NormalizeAngle(Vector3 euler)
        {
            euler.x = NormalizeAngle(euler.x);
            euler.y = NormalizeAngle(euler.y);
            euler.z = NormalizeAngle(euler.z);
            return euler;
        }
        public static float NormalizeAngle(float angle)
        {
            angle = angle % 360;
            if (angle > 180)
                angle -= 360;
            else if (angle < -180)
                angle += 360;
            return angle;
        }

        public static string ReadTextFile(string filePath)
        {
            if (File.Exists(filePath))
            {
                string fileContent = File.ReadAllText(filePath);
                return fileContent;
            }
            else
            {
                Debug.LogError("File does not exist at path: " + filePath);
            }
            return "";
        }

        public static string GetCurrentPath()
        {
            string currentPath;

            if (Application.isEditor)
            {
                // Unity Editor에서 Assets 폴더의 경로
                currentPath = Application.dataPath;
            }
            else
            {
                // 빌드된 실행 파일 경로
                currentPath = Path.GetDirectoryName(Application.dataPath);
            }
            return currentPath;
        }
    }
}