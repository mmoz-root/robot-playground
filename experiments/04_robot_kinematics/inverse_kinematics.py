import numpy as np
import matplotlib.pyplot as plt

LINK_1_LENGTH_MM = 100.0
LINK_2_LENGTH_MM = 50.0

TARGET_X_MM = 100.0
TARGET_Y_MM = 50.0


def forward_kinematics_end_effector(
    link_1_length,
    link_2_length,
    theta_1_degrees,
    theta_2_degrees,
):
    theta_1 = np.deg2rad(theta_1_degrees)
    theta_2 = np.deg2rad(theta_2_degrees)

    end_x = (
        link_1_length * np.cos(theta_1)
        + link_2_length * np.cos(theta_1 + theta_2)
    )

    end_y = (
        link_1_length * np.sin(theta_1)
        + link_2_length * np.sin(theta_1 + theta_2)
    )

    return np.array([end_x, end_y])


def inverse_kinematics(
    link_1_length, link_2_length,
    target_x, target_y
):
    target_distance = np.hypot(target_x, target_y)

    min_reach = abs(link_1_length - link_2_length)
    max_reach = link_1_length + link_2_length

    if not min_reach <= target_distance <= max_reach:
        return target_distance, []


    cos_theta_2 = (
        target_x**2 + target_y**2 - link_1_length**2 - link_2_length**2
    ) / (
        2.0 * link_1_length * link_2_length
    )

    cos_theta_2 = np.clip(cos_theta_2, -1.0, 1.0)

    theta_2_magnitude = np.arccos(cos_theta_2)

    theta_2_options = [
        theta_2_magnitude,
        -theta_2_magnitude
    ]

    solutions = []

    for theta_2 in theta_2_options:
        theta_1 = (
            np.arctan2(target_y, target_x)
            - np.arctan2(
                link_2_length * np.sin(theta_2),
                link_1_length + link_2_length * np.cos(theta_2)
            )
        )

        solutions.append(
            (np.rad2deg(theta_1), np.rad2deg(theta_2))
        )
        

    return target_distance, solutions


distance, solutions = inverse_kinematics(
    LINK_1_LENGTH_MM, LINK_2_LENGTH_MM,
    TARGET_X_MM, TARGET_Y_MM
)

print(f"Target: [{TARGET_X_MM}, {TARGET_Y_MM}] mm")
print(f"Distance: {distance:.3f} mm")

if not solutions:
    print("Target is unreachable.")
else:
    target = np.array([
        TARGET_X_MM,
        TARGET_Y_MM,
    ])

    for index, (theta_1, theta_2) in enumerate(
        solutions,
        start=1,
    ):
        print(
            f"Solution {index}: "
            f"theta1={theta_1:.3f}°, "
            f"theta2={theta_2:.3f}°"
        )

        verified_position = forward_kinematics_end_effector(
            LINK_1_LENGTH_MM,
            LINK_2_LENGTH_MM,
            theta_1,
            theta_2,
        )

        verification_error = np.linalg.norm(
            verified_position - target
        )

        print(f"  FK position: {verified_position}")
        print(
            f"  Verification error: "
            f"{verification_error:.6f} mm"
        )


if solutions:
    figure, axes = plt.subplots(figsize=(7, 7))

    colors = ["tab:blue", "tab:orange"]

    for index, ((theta_1, theta_2), color) in enumerate(
        zip(solutions, colors),
        start=1,
    ):
        theta_1_radians = np.deg2rad(theta_1)

        elbow = np.array([
            LINK_1_LENGTH_MM * np.cos(theta_1_radians),
            LINK_1_LENGTH_MM * np.sin(theta_1_radians),
        ])

        end_effector = forward_kinematics_end_effector(
            LINK_1_LENGTH_MM,
            LINK_2_LENGTH_MM,
            theta_1,
            theta_2,
        )

        axes.plot(
            [0.0, elbow[0], end_effector[0]],
            [0.0, elbow[1], end_effector[1]],
            marker="o",
            linewidth=4,
            label=f"Solution {index}",
            color=color,
        )

    axes.scatter(
        TARGET_X_MM,
        TARGET_Y_MM,
        marker="x",
        s=150,
        color="red",
        label="Target",
        zorder=5,
    )

    plot_limit = (
        LINK_1_LENGTH_MM
        + LINK_2_LENGTH_MM
        + 20.0
    )

    axes.set_xlim(-plot_limit, plot_limit)
    axes.set_ylim(-plot_limit, plot_limit)
    axes.set_aspect("equal", adjustable="box")
    axes.axhline(0.0, color="gray", linewidth=1)
    axes.axvline(0.0, color="gray", linewidth=1)
    axes.grid(True)
    axes.legend()
    axes.set_xlabel("X position (mm)")
    axes.set_ylabel("Y position (mm)")
    axes.set_title("Two-Link Arm — Inverse Kinematics")

    plt.show()