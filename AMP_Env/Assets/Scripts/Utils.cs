using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AMP
{
    public class Utils
    {

        public static string ReadTextFile(Object file)
        {
            if(file == null)
            {
                Debug.LogWarning("File not assigned!");
                return "";
            }

            string filePath = AssetDatabase.GetAssetPath(file);

            return ReadTextFile(filePath);
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