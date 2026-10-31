"""Phonon Vision — Blender addon for phonopy mode visualization.

Registers PropertyGroup, operators, panels, and the frame-change handler.
The bundled phonon-core wheel provides all the math; this addon is the
Blender front-end.
"""

bl_info = {
    "name": "Phonon Vision",
    "author": "Melchor",
    "version": (0, 8, 8),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Phonon",
    "description": "Visualize phonopy phonon modes with publication-quality renders",
    "category": "3D View",
}

import bpy

from . import state, operators, ui, frame_handler


def register():
    # PVSpeciesEntry must register BEFORE PhononVisionProps (the latter refers
    # to it via CollectionProperty(type=PVSpeciesEntry))
    bpy.utils.register_class(state.PVSpeciesEntry)
    bpy.utils.register_class(state.PhononVisionProps)
    bpy.types.Scene.phonon_vision = bpy.props.PointerProperty(type=state.PhononVisionProps)

    operators.register()
    ui.register()
    frame_handler.register_handler()


def unregister():
    frame_handler.unregister_handler()
    ui.unregister()
    operators.unregister()

    del bpy.types.Scene.phonon_vision
    bpy.utils.unregister_class(state.PhononVisionProps)
    bpy.utils.unregister_class(state.PVSpeciesEntry)
