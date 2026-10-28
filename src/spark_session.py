"""
Spark Session Manager for WaterWise.
Configures local Apache PySpark environment with appropriate JVM options,
Java 17 runtime, and Hadoop Windows binaries.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession


def configure_environment() -> None:
    """Detect and configure JAVA_HOME, HADOOP_HOME, and PATH."""
    project_root = Path(__file__).resolve().parent.parent

    # 1. Configure HADOOP_HOME for Windows winutils
    hadoop_dir = project_root / "hadoop_home"
    if hadoop_dir.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        bin_dir = str(hadoop_dir / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    # 2. Configure JAVA_HOME if not already set or invalid
    current_java = os.environ.get("JAVA_HOME", "")
    java_valid = False
    if current_java:
        java_exe = Path(current_java) / "bin" / ("java.exe" if sys.platform == "win32" else "java")
        if java_exe.exists():
            java_valid = True

    if not java_valid:
        candidates = [
            Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"),
            Path(r"C:\Program Files\Eclipse Adoptium\jdk-17"),
            Path(r"C:\Program Files\Java\jdk-17"),
            Path(r"C:\Program Files\Java\jdk-21"),
        ]
        # Also check for any jdk-17 in Eclipse Adoptium directory
        adoptium_dir = Path(r"C:\Program Files\Eclipse Adoptium")
        if adoptium_dir.exists():
            for child in adoptium_dir.iterdir():
                if child.is_dir() and "jdk" in child.name.lower():
                    candidates.insert(0, child)

        for candidate in candidates:
            cand_exe = candidate / "bin" / ("java.exe" if sys.platform == "win32" else "java")
            if cand_exe.exists():
                os.environ["JAVA_HOME"] = str(candidate)
                bin_dir = str(candidate / "bin")
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                break

    # 3. Add JVM modular options required for Java 17+ with Spark
    java_opts = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "-Djava.security.manager=allow"
    )
    existing_opts = os.environ.get("_JAVA_OPTIONS", "")
    if "--add-opens=java.base/java.lang=ALL-UNNAMED" not in existing_opts:
        os.environ["_JAVA_OPTIONS"] = (existing_opts + " " + java_opts).strip()


_spark_instance: Optional[SparkSession] = None


def get_spark_session(app_name: str = "WaterWise", memory: str = "4g") -> SparkSession:
    """
    Get or create a local SparkSession with optimized configuration for local execution.
    """
    global _spark_instance
    if _spark_instance is not None:
        try:
            # Check if existing session is alive
            if not _spark_instance.sparkContext._jsc.sc().isStopped():
                return _spark_instance
        except Exception:
            _spark_instance = None

    configure_environment()

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )

    _spark_instance = builder.getOrCreate()
    _spark_instance.sparkContext.setLogLevel("WARN")
    return _spark_instance


def stop_spark_session() -> None:
    """Stop the running SparkSession."""
    global _spark_instance
    if _spark_instance is not None:
        try:
            _spark_instance.stop()
        except Exception:
            pass
        _spark_instance = None
