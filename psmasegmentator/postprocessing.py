import SimpleITK as sitk
import os
import numpy as np
import totalsegmentator

def expand_segmentation(predicted_image, pet_image, 
                             ct_image_dir = str, suv_threshold=3):
    """
    Expand the predicted segmentation based on the PET image and a given SUV threshold.

    :param predicted_image: SimpleITK image of the predicted segmentation.
    :param pet_image: SimpleITK image of the PET image.
    :param ct_image: SimpleITK image of the CT image.
    :param suv_threshold: The SUV threshold to expand the segmentation by. Defaults to three. 
    :return: The expanded segmentation.
    """
    # Ensure the PET image is of type sitkFloat32
    pet_image = sitk.Cast(pet_image, sitk.sitkFloat32)

    # Label the connected components in the predicted segmentation
    labeled_image = sitk.ConnectedComponent(predicted_image)

    # Get the number of connected components
    num_components = int(sitk.GetArrayViewFromImage(labeled_image).max())

    # Create an empty image to store the expanded segmentation
    expanded_seg = sitk.Image(predicted_image.GetSize(), sitk.sitkUInt8)
    expanded_seg.CopyInformation(predicted_image)

    #Run the TotalSegmentator on the input CT image
    ct_segmentation_nib = totalsegmentator(ct_image_dir,
                                       fastest = True)
    
    #Convert the nibabel image to a NumPy array
    ct_segmentation_array = ct_segmentation_nib.get_fdata()

    #Convert to int
    ct_segmentation_array = ct_segmentation_array.astype(int)

    #Just get the components of the CT segmentation equal to 5 or 21 or 22 (liver, bladder, prostate)
    ct_bladder_prostate = np.isin(ct_segmentation_array, [5, 21, 22]).astype(np.uint8)
    ct_bladder_prostate = np.transpose(ct_bladder_prostate, (2, 1, 0)) #Converting from (z, y, x) to (x, y, z) format


    # Iterate over each connected component
    for i in range(1, num_components + 1):
        # Get the current component
        component = labeled_image == i

        # Find all seed points for the current connected component
        component_array = sitk.GetArrayFromImage(component)
        pet_array = sitk.GetArrayFromImage(pet_image)

        # Find the voxel with the maximum SUV value within the component
        component_voxels = np.argwhere(component_array)
        max_voxel = max(component_voxels, key=lambda x: pet_array[tuple(x)])
        seed_point = tuple(max_voxel[::-1])  # Convert to (z, y, x) format
        seed_point = (int(seed_point[0]), int(seed_point[1]), int(seed_point[2]))
        expanded_component = sitk.ConnectedThreshold(image1=pet_image,
                                             seedList=[seed_point],
                                             lower=suv_threshold,
                                             upper=1000.0,
                                             replaceValue=1)
        
        #Get the array from the expanded component
        expanded_component_array = sitk.GetArrayFromImage(expanded_component)

        #Implement the rubric for selecting which components are expanded. Check if the expanded component
        #just created overlaps with the urinary bladder of the ct segmentation (value is 21), or if the expanded
        #component overlaps with the prostate (value is 22). If it does, then don't expand the component.
        #If it doesn't, then expand the component.
        
        #If any intersection between the expanded component and the liver/bladder/prostate is found, 
        # then don't expand the component.
        if np.any(np.logical_and(expanded_component_array, ct_bladder_prostate)):
            expanded_seg = sitk.Or(expanded_seg, component)
        
        #Handle the case where the component is all zeroes because SUVmax lower than threshold
        elif np.all(expanded_component_array == 0):
            expanded_seg = sitk.Or(expanded_seg, component)
        
        else:
            expanded_seg = sitk.Or(expanded_seg, expanded_component)
            expanded_seg = sitk.Or(expanded_seg, component)
        
    return expanded_seg