using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.TerrainTools;

namespace AMP
{
    [CustomEditor(typeof(CMUSkeleton))]
    public class CMUSkeletonEditor : Editor
    {

        public override void OnInspectorGUI()
        {
            CMUSkeleton c = (CMUSkeleton)target;

            DrawDefaultInspector();

            if (GUILayout.Button("Generate Skeleton", GUILayout.Width(200)))
            {
                c.CreateSkeleton();
            }

        }
    }
}