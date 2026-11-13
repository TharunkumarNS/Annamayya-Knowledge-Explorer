
import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

export type SphereMode = "idle" | "playing" | "searching" | "connections";

type ParticlesProps = {
  mode: SphereMode;
};

function Particles({ mode }: ParticlesProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Generate the particle positions only once.
  const { positions, colors, connections } = useMemo(() => {
    const count = 6000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const connections: number[] = [];

    for (let i = 0; i < count; i++) {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = Math.random() * Math.PI * 2;
      const radius = 2 + (Math.random() - 0.5) * 0.12;

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      const choice = Math.random();
      const color = new THREE.Color(
        choice < 0.55
          ? "#91caff"
          : choice < 0.85
            ? "#ffffff"
            : "#e8bd83"
      );

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;

      // Create small links between particles for graph mode.
      if (i % 12 === 0 && i + 1 < count) {
        connections.push(
          positions[i * 3],
          positions[i * 3 + 1],
          positions[i * 3 + 2],
          positions[(i + 1) * 3],
          positions[(i + 1) * 3 + 1],
          positions[(i + 1) * 3 + 2]
        );
      }
    }

    return {
      positions,
      colors,
      connections: new Float32Array(connections),
    };
  }, []);

  // Change the movement according to the selected mode.
  useFrame((state, delta) => {
    if (!groupRef.current) return;

    const time = state.clock.elapsedTime;

    let speed = 0.12;
    let pulse = 1;
    let tilt = 0.08;

    if (mode === "playing") {
      speed = 0.2;
      pulse = 1 + Math.sin(time * 3) * 0.045;
      tilt = 0.12;
    }

    if (mode === "searching") {
      speed = 0.65;
      pulse = 1 + Math.sin(time * 5) * 0.025;
      tilt = 0.2;
    }

    if (mode === "connections") {
      speed = 0.08;
      pulse = 1 + Math.sin(time * 1.5) * 0.015;
      tilt = 0.3;
    }

    groupRef.current.rotation.y += delta * speed;
    groupRef.current.rotation.x =
      Math.sin(time * 0.3) * tilt;

    groupRef.current.scale.setScalar(pulse);
  });

  return (
    <group ref={groupRef}>
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
          />
        </bufferGeometry>

        <pointsMaterial
          size={0.025}
          vertexColors
          transparent
          opacity={0.95}
          sizeAttenuation
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>

      {/* Connection lines become visible in graph mode. */}
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[connections, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="#70bfff"
          transparent
          opacity={mode === "connections" ? 0.5 : 0.025}
        />
      </lineSegments>
    </group>
  );
}

type ParticleSphereProps = {
  mode?: SphereMode;
};

export default function ParticleSphere({
  mode = "idle",
}: ParticleSphereProps) {
  return (
    <div className="particle-sphere">
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        dpr={[1, 1.5]}
        gl={{ alpha: true, antialias: true }}
      >
        <ambientLight intensity={1} />
        <Particles mode={mode} />
      </Canvas>
    </div>
  );
}
