import removeBackground
import im2Pencil
import SimoSerra

if __name__ == '__main__':

    removeBackground.removeAllBackground(InputImages=r"E:\fypMID1\Results\images",InputAttr=r"E:\fypMID1\Results\attributes",OutputFolder=r"E:\fypMID1\Results\RemovedBackgrounds")
    print()
    im2Pencil.ConvertAllToSketch(InputImages=r"E:\fypMID1\Results\RemovedBackgrounds",OutputFolder=r"E:\fypMID1\Results\Sketches")
    print()
    SimoSerra.SimiplifySketches(InputFolder=r"E:\fypMID1\Results\Sketches",OutputFolder=r"E:\fypMID1\Results\SimplifiedSketch")
