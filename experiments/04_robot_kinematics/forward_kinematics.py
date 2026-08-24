import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

LINK_1_LENGTH_MM = 100.0
LINK_2_LENGTH_MM = 50.0

THETA_1_DEG = 0.0
THETA_2_DEG = 90.0

def forward_kinematics(
    link_1_length, link_2_length,
    theta_1_degrees, theta_2_degrees
):
    theta_1 = np.deg2rad(theta_1_degrees)
    theta_2 = np.deg2rad(theta_2_degrees)

    base = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )

    elbow = np.array(
        [
            link_1_length * np.cos(theta_1),
            link_1_length * np.sin(theta_1)
        ],
        dtype=np.float64
    )

    link_2_world_angle = theta_1 + theta_2

    end_effector = elbow + np.array(
        [
            link_2_length * np.cos(link_2_world_angle),
            link_2_length * np.sin(link_2_world_angle)
        ],
        dtype=np.float64
    )

    return base, elbow, end_effector

base, elbow, end_effector = forward_kinematics(
    LINK_1_LENGTH_MM, LINK_2_LENGTH_MM,
    THETA_1_DEG, THETA_2_DEG
)

np.set_printoptions(
    precision=3,
    suppress=True
)

print(f"Base: {base} mm")
print(f"Elbow: {elbow} mm")
print(f"End-effector: {end_effector} mm")


figure, axes = plt.subplots(
    figsize=(7, 7),
)

figure.subplots_adjust(
    bottom=0.25,
)


arm_line, = axes.plot(
    [],
    [],
    marker="o",
    linewidth=4,
    markersize=10,
)

elbow_label = axes.text(
    0.0,
    0.0,
    "Elbow",
)

end_effector_label = axes.text(
    0.0,
    0.0,
    "End effector",
)

position_text = axes.text(
    0.02,
    0.96,
    "",
    transform=axes.transAxes,
    verticalalignment="top",
)


maximum_reach = (
    LINK_1_LENGTH_MM
    + LINK_2_LENGTH_MM
)

plot_margin = 20.0
plot_limit = maximum_reach + plot_margin

axes.set_xlim(
    -plot_limit,
    plot_limit,
)

axes.set_ylim(
    -plot_limit,
    plot_limit,
)

axes.set_aspect(
    "equal",
    adjustable="box",
)

axes.axhline(
    0.0,
    color="gray",
    linewidth=1,
)

axes.axvline(
    0.0,
    color="gray",
    linewidth=1,
)

axes.grid(True)
axes.set_xlabel("X position (mm)")
axes.set_ylabel("Y position (mm)")
axes.set_title("Two-Link Arm — Forward Kinematics")


theta_1_slider_axes = figure.add_axes(
    [0.20, 0.12, 0.65, 0.03]
)

theta_2_slider_axes = figure.add_axes(
    [0.20, 0.06, 0.65, 0.03]
)

theta_1_slider = Slider(
    theta_1_slider_axes,
    "Theta 1",
    -180.0,
    180.0,
    valinit=THETA_1_DEG,
    valstep=1.0,
)

theta_2_slider = Slider(
    theta_2_slider_axes,
    "Theta 2",
    -180.0,
    180.0,
    valinit=THETA_2_DEG,
    valstep=1.0,
)


def update_arm(_):
    current_base, current_elbow, current_end = (
        forward_kinematics(
            LINK_1_LENGTH_MM,
            LINK_2_LENGTH_MM,
            theta_1_slider.val,
            theta_2_slider.val,
        )
    )

    arm_line.set_data(
        [
            current_base[0],
            current_elbow[0],
            current_end[0],
        ],
        [
            current_base[1],
            current_elbow[1],
            current_end[1],
        ],
    )

    elbow_label.set_position(
        current_elbow + np.array([5.0, 5.0])
    )

    end_effector_label.set_position(
        current_end + np.array([5.0, 5.0])
    )

    position_text.set_text(
        (
            f"End effector: "
            f"({current_end[0]:.1f}, "
            f"{current_end[1]:.1f}) mm"
        )
    )

    figure.canvas.draw_idle()


theta_1_slider.on_changed(update_arm)
theta_2_slider.on_changed(update_arm)

update_arm(None)

plt.show()